use crate::api::OllamaClient;
use crate::parakeet::{self, Parakeet, ParakeetModel};
use crate::whistle::{self, Whistle};
use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use gtk::glib;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub(crate) const TARGET_RATE: u32 = 16_000;
const VAD_FRAME: usize = 512;
const SPEECH_START_THRESHOLD: f32 = 0.5;
const SPEECH_CONTINUE_THRESHOLD: f32 = 0.35;
const PARTIAL_SAMPLES: usize = 24_000; // 1.5 s
const MAX_UTTERANCE_SAMPLES: usize = 30 * TARGET_RATE as usize;
const PRE_ROLL_FRAMES: usize = 8;

#[derive(Debug, Clone)]
pub enum SttEvent {
    SpeechStarted,
    Partial(String),
    Final(String),
    Timeout,
    Error(String),
    Stopped,
}

pub enum PreparedBackend {
    Parakeet(parakeet::SharedParakeet),
    Whistle(Arc<Mutex<Option<Whistle>>>),
    Builtin { client: OllamaClient, model: String },
}

pub struct SttSession {
    _stream: cpal::Stream,
    stop: Arc<AtomicBool>,
}

struct WorkerConfig {
    source_rate: u32,
    language: Option<String>,
    start_timeout: Duration,
    end_silence: Duration,
    continuous: bool,
    /// End the session after this much silence between utterances. Used by
    /// wake-word sessions so the microphone is released until the next wake.
    idle_timeout: Option<Duration>,
    /// While this flag is set the idle timer keeps being pushed back, so a
    /// wake-triggered session is not cut off while a reply is being spoken.
    keep_alive: Option<Arc<AtomicBool>>,
}

impl SttSession {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

pub async fn prepare_whistle(engine: Arc<Mutex<Option<Whistle>>>) -> Result<()> {
    std::fs::create_dir_all(whistle::model_dir())
        .with_context(|| format!("creating {}", whistle::model_dir().display()))?;
    whistle::ensure_downloaded().await?;

    tokio::task::spawn_blocking(move || {
        let mut guard = engine
            .lock()
            .map_err(|_| anyhow::anyhow!("Whistle engine lock failed"))?;
        if guard.is_none() {
            *guard = Some(Whistle::load(&whistle::model_dir())?);
        }
        Ok::<(), anyhow::Error>(())
    })
    .await??;
    Ok(())
}

pub async fn prepare_parakeet(
    engine: parakeet::SharedParakeet,
    model: ParakeetModel,
) -> Result<()> {
    let dir = parakeet::model_dir(model);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    parakeet::ensure_downloaded(model).await?;

    tokio::task::spawn_blocking(move || {
        let mut guard = engine
            .lock()
            .map_err(|_| anyhow::anyhow!("Parakeet engine lock failed"))?;
        if guard.as_ref().is_none_or(|(loaded, _)| *loaded != model) {
            *guard = Some((model, Parakeet::load(&parakeet::model_dir(model))?));
        }
        Ok::<(), anyhow::Error>(())
    })
    .await??;
    Ok(())
}

/// Timing and lifetime knobs for one dictation or live session.
pub struct ListenConfig {
    pub language: Option<String>,
    pub start_timeout: Duration,
    pub end_silence: Duration,
    pub continuous: bool,
    /// End the session after this much silence. Used by wake-word sessions.
    pub idle_timeout: Option<Duration>,
    /// While set, the idle timer keeps being pushed back.
    pub keep_alive: Option<Arc<AtomicBool>>,
}

pub fn start_listening(
    runtime: &tokio::runtime::Runtime,
    backend: PreparedBackend,
    config: ListenConfig,
) -> Result<(SttSession, async_channel::Receiver<SttEvent>)> {
    let (event_tx, event_rx) = async_channel::unbounded::<SttEvent>();

    let error_events = event_tx.clone();
    let capture = start_capture(move |message| {
        let events = error_events.clone();
        glib::MainContext::default().invoke(move || {
            let _ = events.try_send(SttEvent::Error(format!("microphone error: {message}")));
        });
    })?;
    let sample_rate = capture.sample_rate;
    let audio_rx = capture.audio;
    let stop = Arc::new(AtomicBool::new(false));

    let worker_stop = Arc::clone(&stop);
    runtime.spawn(async move {
        if let Err(error) = run_worker(
            audio_rx,
            backend,
            WorkerConfig {
                source_rate: sample_rate,
                language: config.language,
                start_timeout: config.start_timeout,
                end_silence: config.end_silence,
                continuous: config.continuous,
                idle_timeout: config.idle_timeout,
                keep_alive: config.keep_alive,
            },
            event_tx.clone(),
            worker_stop,
        )
        .await
        {
            let _ = event_tx
                .send(SttEvent::Error(format!(
                    "speech recognition failed: {error:#}"
                )))
                .await;
        }
    });

    Ok((
        SttSession {
            _stream: capture.stream,
            stop,
        },
        event_rx,
    ))
}

/// A live microphone stream that delivers mono `f32` chunks at the device rate.
pub struct AudioCapture {
    pub stream: cpal::Stream,
    pub sample_rate: u32,
    pub audio: async_channel::Receiver<Vec<f32>>,
}

/// Prefer 48 kHz so the audio server does not resample every block. The
/// channel count stays at the device default, which keeps the capture format
/// compatible with the graph and avoids needless mixing.
fn capture_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig, anyhow::Error> {
    let default = device
        .default_input_config()
        .context("no default microphone configuration")?;
    if default.sample_rate().0 == 48_000 {
        return Ok(default);
    }
    if let Ok(configs) = device.supported_input_configs() {
        for config in configs {
            if config.channels() != default.channels()
                || config.sample_format() != default.sample_format()
                || config.min_sample_rate().0 > 48_000
                || config.max_sample_rate().0 < 48_000
            {
                continue;
            }
            return Ok(config.with_sample_rate(cpal::SampleRate(48_000)));
        }
    }
    Ok(default)
}

/// Open the default microphone and start streaming mixed-down samples.
pub fn start_capture(on_error: impl Fn(String) + Send + Sync + 'static) -> Result<AudioCapture> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no microphone was found")?;
    let supported = capture_config(&device)?;
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let config = supported.config();

    let (audio_tx, audio_rx) = async_channel::unbounded::<Vec<f32>>();
    let on_error = Arc::new(on_error);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let sender = audio_tx.clone();
            let on_error = Arc::clone(&on_error);
            device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    let _ = sender.try_send(mix_f32(data, channels));
                },
                move |error| on_error(error.to_string()),
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let sender = audio_tx.clone();
            let on_error = Arc::clone(&on_error);
            device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    let samples: Vec<f32> = data
                        .iter()
                        .map(|sample| *sample as f32 / 32_768.0)
                        .collect();
                    let _ = sender.try_send(mix_f32(&samples, channels));
                },
                move |error| on_error(error.to_string()),
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let sender = audio_tx.clone();
            let on_error = Arc::clone(&on_error);
            device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    let samples: Vec<f32> = data
                        .iter()
                        .map(|sample| (*sample as f32 - 32_768.0) / 32_768.0)
                        .collect();
                    let _ = sender.try_send(mix_f32(&samples, channels));
                },
                move |error| on_error(error.to_string()),
                None,
            )?
        }
        other => bail!("unsupported microphone sample format: {other:?}"),
    };

    stream.play().context("starting microphone")?;

    Ok(AudioCapture {
        stream,
        sample_rate,
        audio: audio_rx,
    })
}

fn mix_f32(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

pub(crate) struct LinearResampler {
    step: f64,
    position: f64,
    tail: Vec<f32>,
}

impl LinearResampler {
    pub(crate) fn new(source_rate: u32) -> Self {
        Self {
            step: source_rate as f64 / TARGET_RATE as f64,
            position: 0.0,
            tail: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, input: &[f32]) -> Vec<f32> {
        self.tail.extend_from_slice(input);
        let mut output = Vec::new();

        while self.position + 1.0 < self.tail.len() as f64 {
            let index = self.position.floor() as usize;
            let fraction = self.position - index as f64;
            let first = self.tail[index];
            let second = self.tail[index + 1];
            output.push(first + (second - first) * fraction as f32);
            self.position += self.step;
        }

        let drop = (self.position.floor() as usize).min(self.tail.len().saturating_sub(1));
        if drop > 0 {
            self.tail.drain(..drop);
            self.position -= drop as f64;
        }
        output
    }
}

async fn run_worker(
    audio_rx: async_channel::Receiver<Vec<f32>>,
    backend: PreparedBackend,
    config: WorkerConfig,
    events: async_channel::Sender<SttEvent>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let WorkerConfig {
        source_rate,
        language,
        start_timeout,
        end_silence,
        continuous,
        idle_timeout,
        keep_alive,
    } = config;
    let mut vad = silero::Session::bundled().context("loading Silero VAD")?;
    let mut vad_state = silero::StreamState::new(silero::SampleRate::Rate16k);
    let mut resampler = LinearResampler::new(source_rate);
    let mut pending: Vec<f32> = Vec::new();
    let mut pre_roll: VecDeque<f32> = VecDeque::new();
    let mut speech: Vec<f32> = Vec::new();
    let mut speaking = false;
    let mut silence_samples = 0usize;
    let mut total_samples = 0usize;
    let mut last_partial = 0usize;
    let mut terminal = false;
    let started = std::time::Instant::now();
    let mut last_activity = started;

    let start_timeout_samples = (start_timeout.as_secs_f64() * TARGET_RATE as f64) as usize;
    let end_silence_samples = (end_silence.as_secs_f64() * TARGET_RATE as f64) as usize;

    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        if keep_alive
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            last_activity = std::time::Instant::now();
        }

        let chunk = match tokio::time::timeout(Duration::from_millis(100), audio_rx.recv()).await {
            Ok(Ok(chunk)) => chunk,
            Ok(Err(_)) => break,
            Err(_) => {
                let idle_due =
                    idle_timeout.is_some_and(|idle| !speaking && last_activity.elapsed() >= idle);
                let start_due = idle_timeout.is_none()
                    && !continuous
                    && !speaking
                    && started.elapsed() >= start_timeout;
                if idle_due || start_due {
                    let _ = events.send(SttEvent::Timeout).await;
                    terminal = true;
                    break;
                }
                continue;
            }
        };

        let resampled = resampler.push(&chunk);
        for sample in resampled {
            pending.push(sample);
            if pending.len() < VAD_FRAME {
                continue;
            }
            let frame: Vec<f32> = pending.drain(..VAD_FRAME).collect();
            total_samples += VAD_FRAME;
            let probability = vad
                .infer_chunk(&mut vad_state, &frame)
                .context("running Silero VAD")?;

            if !speaking {
                for sample in &frame {
                    pre_roll.push_back(*sample);
                }
                while pre_roll.len() > VAD_FRAME * PRE_ROLL_FRAMES {
                    pre_roll.pop_front();
                }

                if probability >= SPEECH_START_THRESHOLD {
                    speaking = true;
                    speech.clear();
                    speech.extend(pre_roll.drain(..));
                    silence_samples = 0;
                    last_partial = speech.len();
                    last_activity = std::time::Instant::now();
                    let _ = events.send(SttEvent::SpeechStarted).await;
                } else {
                    let timed_out = match idle_timeout {
                        Some(idle) => last_activity.elapsed() >= idle,
                        None => !continuous && total_samples >= start_timeout_samples,
                    };
                    if timed_out {
                        let _ = events.send(SttEvent::Timeout).await;
                        terminal = true;
                        break;
                    }
                }
                continue;
            }

            speech.extend_from_slice(&frame);
            if probability < SPEECH_CONTINUE_THRESHOLD {
                silence_samples += VAD_FRAME;
            } else {
                silence_samples = 0;
            }

            if speech.len() >= PARTIAL_SAMPLES
                && speech.len() - last_partial >= PARTIAL_SAMPLES
                && speech.len() < MAX_UTTERANCE_SAMPLES
            {
                match transcribe_pcm(&backend, &speech, language.as_deref()).await {
                    Ok(text) => {
                        last_partial = speech.len();
                        if !text.trim().is_empty() {
                            let _ = events.send(SttEvent::Partial(text)).await;
                        }
                    }
                    Err(error) => {
                        tracing::warn!("partial speech recognition failed: {error:#}");
                    }
                }
            }

            if silence_samples >= end_silence_samples || speech.len() >= MAX_UTTERANCE_SAMPLES {
                match transcribe_pcm(&backend, &speech, language.as_deref()).await {
                    Ok(text) => {
                        let _ = events.send(SttEvent::Final(text)).await;
                    }
                    Err(error) => {
                        let _ = events
                            .send(SttEvent::Error(format!(
                                "speech recognition failed: {error:#}"
                            )))
                            .await;
                    }
                }
                if continuous {
                    speaking = false;
                    speech.clear();
                    silence_samples = 0;
                    last_partial = 0;
                    pre_roll.clear();
                    last_activity = std::time::Instant::now();
                } else {
                    terminal = true;
                    break;
                }
            }
        }
    }

    if !terminal {
        if speaking && !speech.is_empty() {
            match transcribe_pcm(&backend, &speech, language.as_deref()).await {
                Ok(text) => {
                    let _ = events.send(SttEvent::Final(text)).await;
                }
                Err(error) => {
                    let _ = events
                        .send(SttEvent::Error(format!(
                            "speech recognition failed: {error:#}"
                        )))
                        .await;
                }
            }
        } else {
            let _ = events.send(SttEvent::Stopped).await;
        }
    }

    Ok(())
}

async fn transcribe_pcm(
    backend: &PreparedBackend,
    pcm: &[f32],
    language: Option<&str>,
) -> Result<String> {
    match backend {
        PreparedBackend::Parakeet(engine) => {
            let engine = Arc::clone(engine);
            let pcm = pcm.to_vec();
            tokio::task::spawn_blocking(move || {
                let mut guard = engine
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Parakeet engine lock failed"))?;
                let (_, engine) = guard
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("Parakeet engine is not loaded"))?;
                engine.transcribe(&pcm)
            })
            .await?
        }
        PreparedBackend::Whistle(engine) => {
            let engine = Arc::clone(engine);
            let pcm = pcm.to_vec();
            let language = language.map(str::to_string);
            tokio::task::spawn_blocking(move || {
                let mut guard = engine
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Whistle engine lock failed"))?;
                let engine = guard
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("Whistle engine is not loaded"))?;
                engine
                    .transcribe(&pcm, language.as_deref())
                    .map(|result| result.text)
            })
            .await?
        }
        PreparedBackend::Builtin { client, model } => {
            let wav = encode_wav(pcm)?;
            client.transcribe(model, wav, language).await
        }
    }
}

fn encode_wav(pcm: &[f32]) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut bytes = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut bytes);
        let mut writer = hound::WavWriter::new(cursor, spec)?;
        for sample in pcm {
            let value = (sample.clamp(-1.0, 1.0) * 32_767.0) as i16;
            writer.write_sample(value)?;
        }
        writer.finalize()?;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{encode_wav, run_worker, LinearResampler};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn resampler_preserves_duration() {
        let mut resampler = LinearResampler::new(48_000);
        let input = vec![0.25f32; 48_000];
        let output = resampler.push(&input);
        assert!((output.len() as i64 - 16_000).abs() < 32);
    }

    #[test]
    fn resampler_handles_odd_chunk_sizes() {
        let mut resampler = LinearResampler::new(44_100);
        let mut total = 0;
        for chunk in [vec![0.1f32; 1024], vec![0.2f32; 1023], vec![0.3f32; 1021]] {
            total += resampler.push(&chunk).len();
        }
        assert!(total > 1_000);
    }

    #[test]
    fn wav_has_riff_header() {
        let wav = encode_wav(&vec![0.0f32; 1600]).expect("wav");
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }

    #[test]
    #[ignore = "requires the Whistle ONNX files and a test WAV"]
    fn worker_transcribes_test_wav() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let samples = read_test_wav();
            let engine = Arc::new(std::sync::Mutex::new(Some(
                crate::whistle::Whistle::load(std::path::Path::new("/tmp/opencode/whistle-model"))
                    .expect("load Whistle"),
            )));
            let (audio_tx, audio_rx) = async_channel::unbounded();
            for chunk in samples.chunks(1024) {
                audio_tx.send(chunk.to_vec()).await.unwrap();
            }
            drop(audio_tx);

            let (events_tx, events_rx) = async_channel::unbounded();
            let worker = tokio::spawn(run_worker(
                audio_rx,
                super::PreparedBackend::Whistle(engine),
                super::WorkerConfig {
                    source_rate: 16_000,
                    language: Some("en".to_string()),
                    start_timeout: Duration::from_secs(5),
                    end_silence: Duration::from_millis(700),
                    continuous: false,
                    idle_timeout: None,
                    keep_alive: None,
                },
                events_tx,
                Arc::new(AtomicBool::new(false)),
            ));

            let mut final_text = None;
            while let Ok(event) = events_rx.recv().await {
                match event {
                    super::SttEvent::Final(text) => {
                        final_text = Some(text);
                        break;
                    }
                    super::SttEvent::Error(error) => panic!("{error}"),
                    _ => {}
                }
            }
            worker.await.unwrap().unwrap();
            let text = final_text.expect("final transcript");
            println!("worker transcript={text:?}");
            assert!(!text.is_empty());
        });
    }

    #[test]
    #[ignore = "requires a local Ollama model with audio input"]
    fn worker_transcribes_with_builtin_ollama() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let samples = read_test_wav();
            let client = crate::api::OllamaClient::new("http://localhost:11434", "");
            let models = client.list_models().await.expect("list models");
            let model = models
                .iter()
                .find(|model| model.capabilities.iter().any(|item| item == "audio"))
                .map(|model| model.name.clone())
                .expect("no audio-capable model installed");

            let (audio_tx, audio_rx) = async_channel::unbounded();
            for chunk in samples.chunks(1024) {
                audio_tx.send(chunk.to_vec()).await.unwrap();
            }
            drop(audio_tx);

            let (events_tx, events_rx) = async_channel::unbounded();
            let worker = tokio::spawn(run_worker(
                audio_rx,
                super::PreparedBackend::Builtin { client, model },
                super::WorkerConfig {
                    source_rate: 16_000,
                    language: Some("en".to_string()),
                    start_timeout: Duration::from_secs(5),
                    end_silence: Duration::from_millis(700),
                    continuous: false,
                    idle_timeout: None,
                    keep_alive: None,
                },
                events_tx,
                Arc::new(AtomicBool::new(false)),
            ));

            let mut final_text = None;
            while let Ok(event) = events_rx.recv().await {
                match event {
                    super::SttEvent::Final(text) => {
                        final_text = Some(text);
                        break;
                    }
                    super::SttEvent::Error(error) => panic!("{error}"),
                    _ => {}
                }
            }
            worker.await.unwrap().unwrap();
            let text = final_text.expect("final transcript");
            println!("ollama transcript={text:?}");
            assert!(text.to_lowercase().contains("whistle"));
        });
    }

    #[test]
    #[ignore = "requires the Whistle ONNX files and test WAVs"]
    fn continuous_worker_transcribes_two_turns() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut samples = read_wav("/tmp/opencode/stt-mem/s1.wav");
            samples.extend(std::iter::repeat_n(0.0, 16_000));
            samples.extend(read_wav("/tmp/opencode/stt-mem/s3.wav"));
            samples.extend(std::iter::repeat_n(0.0, 16_000));

            let engine = Arc::new(std::sync::Mutex::new(Some(
                crate::whistle::Whistle::load(std::path::Path::new("/tmp/opencode/whistle-model"))
                    .expect("load Whistle"),
            )));
            let (audio_tx, audio_rx) = async_channel::unbounded();
            for chunk in samples.chunks(1024) {
                audio_tx.send(chunk.to_vec()).await.unwrap();
            }
            drop(audio_tx);

            let (events_tx, events_rx) = async_channel::unbounded();
            let worker = tokio::spawn(run_worker(
                audio_rx,
                super::PreparedBackend::Whistle(engine),
                super::WorkerConfig {
                    source_rate: 16_000,
                    language: Some("en".to_string()),
                    start_timeout: Duration::from_secs(5),
                    end_silence: Duration::from_millis(700),
                    continuous: true,
                    idle_timeout: None,
                    keep_alive: None,
                },
                events_tx,
                Arc::new(AtomicBool::new(false)),
            ));

            let mut transcripts = Vec::new();
            while let Ok(event) = events_rx.recv().await {
                match event {
                    super::SttEvent::Final(text) => {
                        println!("continuous turn: {text:?}");
                        transcripts.push(text);
                        if transcripts.len() == 2 {
                            break;
                        }
                    }
                    super::SttEvent::Error(error) => panic!("{error}"),
                    _ => {}
                }
            }
            drop(events_rx);
            worker.abort();
            let _ = worker.await;
            assert_eq!(transcripts.len(), 2);
            assert!(transcripts.iter().all(|text| !text.trim().is_empty()));
        });
    }

    #[test]
    #[ignore = "requires the Parakeet Redux ONNX files and a test WAV"]
    fn worker_transcribes_with_parakeet() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let samples = read_wav("/tmp/opencode/stt-mem/s1.wav");
            let engine = Arc::new(std::sync::Mutex::new(Some((
                crate::parakeet::ParakeetModel::Redux,
                crate::parakeet::Parakeet::load(&crate::parakeet::model_dir(
                    crate::parakeet::ParakeetModel::Redux,
                ))
                .expect("load Parakeet"),
            ))));
            let (audio_tx, audio_rx) = async_channel::unbounded();
            for chunk in samples.chunks(1024) {
                audio_tx.send(chunk.to_vec()).await.unwrap();
            }
            drop(audio_tx);

            let (events_tx, events_rx) = async_channel::unbounded();
            let worker = tokio::spawn(run_worker(
                audio_rx,
                super::PreparedBackend::Parakeet(engine),
                super::WorkerConfig {
                    source_rate: 16_000,
                    language: None,
                    start_timeout: Duration::from_secs(5),
                    end_silence: Duration::from_millis(700),
                    continuous: false,
                    idle_timeout: None,
                    keep_alive: None,
                },
                events_tx,
                Arc::new(AtomicBool::new(false)),
            ));

            let mut final_text = None;
            while let Ok(event) = events_rx.recv().await {
                match event {
                    super::SttEvent::Final(text) => {
                        final_text = Some(text);
                        break;
                    }
                    super::SttEvent::Error(error) => panic!("{error}"),
                    _ => {}
                }
            }
            worker.await.unwrap().unwrap();
            let text = final_text.expect("final transcript");
            println!("parakeet worker transcript={text:?}");
            assert!(text.to_lowercase().contains("quick") || text.to_lowercase().contains("fox"));
        });
    }

    fn read_test_wav() -> Vec<f32> {
        read_wav("/tmp/opencode/whistle-test.wav")
    }

    fn read_wav(path: &str) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path).expect("wav");
        reader
            .samples::<i16>()
            .map(|sample| sample.expect("sample") as f32 / 32_768.0)
            .collect()
    }
}
