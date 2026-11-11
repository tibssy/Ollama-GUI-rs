use anyhow::{bail, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const CONFIG_NAME: &str = "config.json";
const VOCAB_NAME: &str = "vocab.txt";
const PREPROCESSOR_NAME: &str = "preprocessor.onnx";
const ENCODER_NAME: &str = "encoder-model.onnx";
const DECODER_NAME: &str = "decoder_joint-model.onnx";

static DOWNLOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Which Parakeet export the app should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParakeetModel {
    /// Small and fast, intended for CPUs.
    #[default]
    Redux,
    /// Full-size model with better accuracy, but slower on CPU.
    Ultra,
}

impl ParakeetModel {
    pub fn from_setting(value: &str) -> Self {
        if value.eq_ignore_ascii_case("ultra") {
            Self::Ultra
        } else {
            Self::Redux
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Redux => "redux",
            Self::Ultra => "ultra",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Redux => "Parakeet Redux",
            Self::Ultra => "Parakeet Ultra",
        }
    }

    pub fn approx_size(self) -> &'static str {
        match self {
            Self::Redux => "~420 MB",
            Self::Ultra => "~720 MB",
        }
    }

    pub fn dir(self) -> PathBuf {
        crate::config::data_dir().join("models").join(match self {
            Self::Redux => "parakeet-redux",
            Self::Ultra => "parakeet-ultra",
        })
    }

    fn repo(self) -> &'static str {
        match self {
            Self::Redux => "https://huggingface.co/eschmidbauer/parakeet-redux-onnx/resolve/main",
            Self::Ultra => "https://huggingface.co/mrfakename/parakeet-ultra-ONNX/resolve/main",
        }
    }

    fn files(self) -> &'static [FileSpec] {
        match self {
            Self::Redux => REDUX_FILES,
            Self::Ultra => ULTRA_FILES,
        }
    }
}

struct FileSpec {
    remote: &'static str,
    local: &'static str,
    min_bytes: u64,
}

const REDUX_FILES: &[FileSpec] = &[
    FileSpec {
        remote: CONFIG_NAME,
        local: CONFIG_NAME,
        min_bytes: 100,
    },
    FileSpec {
        remote: VOCAB_NAME,
        local: VOCAB_NAME,
        min_bytes: 1_000,
    },
    FileSpec {
        remote: PREPROCESSOR_NAME,
        local: PREPROCESSOR_NAME,
        min_bytes: 100_000,
    },
    FileSpec {
        remote: ENCODER_NAME,
        local: ENCODER_NAME,
        min_bytes: 300_000_000,
    },
    FileSpec {
        remote: DECODER_NAME,
        local: DECODER_NAME,
        min_bytes: 60_000_000,
    },
];

const ULTRA_FILES: &[FileSpec] = &[
    FileSpec {
        remote: CONFIG_NAME,
        local: CONFIG_NAME,
        min_bytes: 100,
    },
    FileSpec {
        remote: VOCAB_NAME,
        local: VOCAB_NAME,
        min_bytes: 50_000,
    },
    FileSpec {
        remote: PREPROCESSOR_NAME,
        local: PREPROCESSOR_NAME,
        min_bytes: 100_000,
    },
    FileSpec {
        // The q8 encoder is the closest quantized export to the fp32 model.
        remote: "encoder-model.q8.onnx",
        local: ENCODER_NAME,
        min_bytes: 600_000_000,
    },
    FileSpec {
        remote: DECODER_NAME,
        local: DECODER_NAME,
        min_bytes: 60_000_000,
    },
];

/// Shared handle to the lazily loaded Parakeet engine and the variant it holds.
pub type SharedParakeet = std::sync::Arc<std::sync::Mutex<Option<(ParakeetModel, Parakeet)>>>;

#[derive(Debug, Deserialize)]
struct Config {
    #[allow(dead_code)]
    features_size: usize,
    #[allow(dead_code)]
    subsampling_factor: usize,
    max_tokens_per_step: usize,
    vocab_size: usize,
    blank_token_id: i32,
    durations: Vec<u32>,
    encoder_frame_seconds: f32,
    sample_rate: u32,
}

pub fn model_dir(model: ParakeetModel) -> PathBuf {
    model.dir()
}

fn path_for(model: ParakeetModel, name: &str) -> PathBuf {
    model_dir(model).join(name)
}

pub fn is_downloaded(model: ParakeetModel) -> bool {
    model.files().iter().all(|file| {
        std::fs::metadata(path_for(model, file.local))
            .map(|meta| meta.len() >= file.min_bytes)
            .unwrap_or(false)
    })
}

pub async fn ensure_downloaded(model: ParakeetModel) -> Result<()> {
    let _guard = DOWNLOAD_LOCK.lock().await;
    if is_downloaded(model) {
        return Ok(());
    }

    let dir = model_dir(model);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(900))
        .build()
        .context("creating Parakeet download client")?;

    for file in model.files() {
        let url = format!("{}/{}", model.repo(), file.remote);
        download(&client, &url, &path_for(model, file.local), file.min_bytes).await?;
    }

    if !is_downloaded(model) {
        bail!(
            "{} model download did not produce complete files",
            model.display_name()
        );
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
    if (bytes.len() as u64) < min_bytes {
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

pub struct Parakeet {
    preprocessor: Session,
    encoder: Session,
    decoder: Session,
    pieces: Vec<String>,
    skip: HashSet<usize>,
    blank_id: usize,
    vocab_size: usize,
    durations: Vec<u32>,
    max_tokens_per_step: usize,
    frame_seconds: f32,
    sample_rate: u32,
    state_shape: [usize; 3],
}

impl Parakeet {
    pub fn load(dir: &Path) -> Result<Self> {
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(dir.join(CONFIG_NAME))
                .with_context(|| format!("reading {}", dir.join(CONFIG_NAME).display()))?,
        )
        .context("parsing Parakeet config.json")?;
        let (pieces, skip) = load_vocab(&dir.join(VOCAB_NAME))?;

        let preprocessor = session(dir.join(PREPROCESSOR_NAME), "preprocessor")?;
        let encoder = session(dir.join(ENCODER_NAME), "encoder")?;
        let decoder = session(dir.join(DECODER_NAME), "decoder")?;
        let state_shape = [2usize, 1, 640];

        Ok(Self {
            preprocessor,
            encoder,
            decoder,
            pieces,
            skip,
            blank_id: config.blank_token_id as usize,
            vocab_size: config.vocab_size,
            durations: config.durations,
            max_tokens_per_step: config.max_tokens_per_step,
            frame_seconds: config.encoder_frame_seconds,
            sample_rate: config.sample_rate,
            state_shape,
        })
    }

    pub fn transcribe(&mut self, pcm: &[f32]) -> Result<String> {
        anyhow::ensure!(!pcm.is_empty(), "audio is empty");
        anyhow::ensure!(self.sample_rate == 16_000, "Parakeet expects 16 kHz audio");

        let (features, feature_shape) = {
            let waveforms = Tensor::from_array(([1usize, pcm.len()], pcm.to_vec()))?;
            let lengths = Tensor::from_array(([1usize], vec![pcm.len() as i64]))?;
            let outputs = self.preprocessor.run(ort::inputs![
                "waveforms" => waveforms,
                "waveforms_lens" => lengths,
            ])?;
            let (shape, data) = outputs["features"].try_extract_tensor::<f32>()?;
            (data.to_vec(), shape_usize(shape))
        };
        let feature_len = feature_shape[2];
        if feature_len == 0 {
            return Ok(String::new());
        }

        let (encoded, encoded_shape) = {
            let audio = Tensor::from_array((feature_shape.clone(), features))?;
            let length = Tensor::from_array(([1usize], vec![feature_len as i64]))?;
            let outputs = self.encoder.run(ort::inputs![
                "audio_signal" => audio,
                "length" => length,
            ])?;
            let (shape, data) = outputs["outputs"].try_extract_tensor::<f32>()?;
            let encoded_lengths = outputs["encoded_lengths"].try_extract_tensor::<i64>()?.1;
            let frames = encoded_lengths[0] as usize;
            let dimensions = shape[1] as usize;
            let stride = shape[2] as usize;
            let mut encoded = vec![0.0f32; frames * dimensions];
            for dimension in 0..dimensions {
                for frame in 0..frames {
                    encoded[frame * dimensions + dimension] = data[dimension * stride + frame];
                }
            }
            (encoded, vec![1usize, dimensions, frames])
        };

        let frames = encoded_shape[2];
        if frames == 0 {
            return Ok(String::new());
        }

        let mut state_len = 1usize;
        for dimension in self.state_shape {
            state_len *= dimension;
        }
        let mut state_1 = vec![0.0f32; state_len];
        let mut state_2 = vec![0.0f32; state_len];
        let mut last_token = self.blank_id as i32;
        let mut frame = 0usize;
        let mut steps = self.max_tokens_per_step.saturating_mul(frames);
        let mut tokens = Vec::new();

        while frame < frames && steps > 0 {
            let encoder_output = &encoded[frame * 1024..(frame + 1) * 1024];
            let (logits, next_state_1, next_state_2) = {
                let encoder_tensor =
                    Tensor::from_array(([1usize, 1024, 1], encoder_output.to_vec()))?;
                let targets = Tensor::from_array(([1usize, 1], vec![last_token]))?;
                let target_length = Tensor::from_array(([1usize], vec![1i32]))?;
                let state_1_tensor = Tensor::from_array((self.state_shape, state_1.clone()))?;
                let state_2_tensor = Tensor::from_array((self.state_shape, state_2.clone()))?;
                let outputs = self.decoder.run(ort::inputs![
                    "encoder_outputs" => encoder_tensor,
                    "targets" => targets,
                    "target_length" => target_length,
                    "input_states_1" => state_1_tensor,
                    "input_states_2" => state_2_tensor,
                ])?;
                let (_, logits) = outputs["outputs"].try_extract_tensor::<f32>()?;
                let logits = logits.to_vec();
                let state_1 = outputs["output_states_1"]
                    .try_extract_tensor::<f32>()?
                    .1
                    .to_vec();
                let state_2 = outputs["output_states_2"]
                    .try_extract_tensor::<f32>()?
                    .1
                    .to_vec();
                (logits, state_1, state_2)
            };

            let token = argmax(&logits[..self.vocab_size]);
            let duration_index = argmax(&logits[self.vocab_size..]);
            let mut duration = *self.durations.get(duration_index).unwrap_or(&0) as usize;
            if token == self.blank_id && duration == 0 {
                duration = 1;
            }

            tokens.push(token);
            frame += duration;
            if token != self.blank_id {
                last_token = token as i32;
                state_1 = next_state_1;
                state_2 = next_state_2;
            }
            steps -= 1;
        }

        Ok(self.decode(&tokens))
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let mut text = String::new();
        for token in tokens {
            if *token == self.blank_id || self.skip.contains(token) {
                continue;
            }
            if let Some(piece) = self.pieces.get(*token) {
                text.push_str(piece);
            }
        }
        let text = text.replace('▁', " ");
        text.trim().to_string()
    }

    #[allow(dead_code)]
    pub fn frame_seconds(&self) -> f32 {
        self.frame_seconds
    }
}

fn session(path: PathBuf, name: &str) -> Result<Session> {
    Session::builder()
        .map_err(|error| anyhow::anyhow!("creating Parakeet {name} session: {error}"))?
        .with_intra_threads(4)
        .map_err(|error| anyhow::anyhow!("configuring Parakeet {name}: {error}"))?
        .commit_from_file(&path)
        .with_context(|| format!("loading {}", path.display()))
}

fn load_vocab(path: &Path) -> Result<(Vec<String>, HashSet<usize>)> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut pieces = Vec::new();
    let mut skip = HashSet::new();

    for line in text.lines() {
        let Some((piece, id)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(id) = id.parse::<usize>() else {
            continue;
        };
        if pieces.len() <= id {
            pieces.resize(id + 1, String::new());
        }
        if matches!(piece, "<unk>" | "<pad>" | "<blk>") {
            skip.insert(id);
        }
        pieces[id] = piece.to_string();
    }

    Ok((pieces, skip))
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate().skip(1) {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

fn shape_usize(shape: &ort::value::Shape) -> Vec<usize> {
    shape.as_ref().iter().map(|dim| *dim as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::{Parakeet, ParakeetModel};

    #[test]
    fn parses_model_names() {
        assert_eq!(ParakeetModel::from_setting("redux"), ParakeetModel::Redux);
        assert_eq!(ParakeetModel::from_setting("ultra"), ParakeetModel::Ultra);
        assert_eq!(ParakeetModel::from_setting("ULTRA"), ParakeetModel::Ultra);
        assert_eq!(ParakeetModel::from_setting(""), ParakeetModel::Redux);
    }

    #[test]
    #[ignore = "downloads the Parakeet Redux model (~418 MB) and requires a test WAV"]
    fn downloads_and_transcribes() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(super::ensure_downloaded(ParakeetModel::Redux))
            .expect("download");
        let mut parakeet = Parakeet::load(&super::model_dir(ParakeetModel::Redux)).expect("load");
        let mut reader = hound::WavReader::open("/tmp/opencode/stt-mem/s1.wav").expect("wav");
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|sample| sample.expect("sample") as f32 / 32_768.0)
            .collect();
        let text = parakeet.transcribe(&samples).expect("transcribe");
        println!("parakeet transcript={text:?}");
        assert!(text.to_lowercase().contains("quick") || text.to_lowercase().contains("fox"));
    }

    #[test]
    #[ignore = "downloads the Parakeet Ultra model (~720 MB) and requires a test WAV"]
    fn ultra_downloads_and_transcribes() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(super::ensure_downloaded(ParakeetModel::Ultra))
            .expect("download");
        let mut parakeet = Parakeet::load(&super::model_dir(ParakeetModel::Ultra)).expect("load");
        let mut reader = hound::WavReader::open("/tmp/opencode/stt-mem/s1.wav").expect("wav");
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|sample| sample.expect("sample") as f32 / 32_768.0)
            .collect();
        let text = parakeet.transcribe(&samples).expect("transcribe");
        println!("parakeet ultra transcript={text:?}");
        assert!(text.to_lowercase().contains("quick") || text.to_lowercase().contains("fox"));
    }

    #[test]
    #[ignore = "requires the Parakeet Ultra ONNX files and a test WAV"]
    fn ultra_prepare_loads_the_shared_engine() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let engine: crate::parakeet::SharedParakeet =
                std::sync::Arc::new(std::sync::Mutex::new(None));
            crate::stt::prepare_parakeet(std::sync::Arc::clone(&engine), ParakeetModel::Ultra)
                .await
                .expect("prepare Ultra");

            let mut reader = hound::WavReader::open("/tmp/opencode/stt-mem/s1.wav").expect("wav");
            let samples: Vec<f32> = reader
                .samples::<i16>()
                .map(|sample| sample.expect("sample") as f32 / 32_768.0)
                .collect();
            let mut guard = engine.lock().expect("lock");
            let (model, parakeet) = guard.as_mut().expect("loaded engine");
            assert_eq!(*model, ParakeetModel::Ultra);
            let text = parakeet.transcribe(&samples).expect("transcribe");
            println!("parakeet ultra shared engine transcript={text:?}");
            assert!(text.to_lowercase().contains("quick") || text.to_lowercase().contains("fox"));
        });
    }
}
