use anyhow::{bail, Context, Result};
use ort::session::Session;
use ort::value::{Tensor, TensorRef};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MODEL_REPO: &str = "https://huggingface.co/mrfakename/whistle-ONNX/resolve/main";
const PACK_NAME: &str = "whistle.pack";
const ENCODER_NAME: &str = "encoder.onnx";
const ENCODER_DATA_NAME: &str = "encoder.onnx.data";
const DECODER_NAME: &str = "decoder.onnx";
const DECODER_DATA_NAME: &str = "decoder.onnx.data";

const MODEL_MIN_BYTES: u64 = 60_000_000;
const PACK_MIN_BYTES: u64 = 15_000_000;

const N_FFT: usize = 512;
const HOP: usize = 160;
const WIN: usize = 400;
const MEL_BINS: usize = 80;
const FFT_BINS: usize = N_FFT / 2 + 1;

const LANG0: i64 = 8192;
const EOS: i64 = 1;
const BOS: i64 = 2;
const TEXT_VOCAB: usize = 8192;
const SLOTS: usize = 18_432;
const SEED: u32 = 0x9E37_79B9;
const PRIME: u32 = 0x0100_0193;
const MAX_ENGRAM_CACHE_ROWS: usize = 16_384;

pub const WHISTLE_LANGUAGES: &[&str] = &["en", "de", "fr", "es", "it", "nl", "pl"];

static DOWNLOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Debug, Deserialize)]
struct Manifest {
    codebooks: HashMap<String, Vec<f32>>,
    engram: Vec<EngramSegment>,
    tok: TokenizerData,
    melfb: MelFilterbank,
}

#[derive(Debug, Deserialize)]
struct EngramSegment {
    src: usize,
    out: usize,
    #[serde(rename = "in")]
    input: usize,
    bits: u8,
    group: usize,
}

#[derive(Debug, Deserialize)]
struct TokenizerData {
    pieces: Vec<String>,
    types: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct MelFilterbank {
    src: usize,
    shape: Vec<usize>,
}

#[derive(Debug)]
struct Pack {
    buf: Vec<u8>,
    base: usize,
    manifest: Manifest,
    melfb: Vec<f32>,
}

#[derive(Default)]
struct EngramCache {
    rows: Vec<HashMap<usize, Arc<[f32]>>>,
}

pub struct Transcription {
    pub text: String,
    #[allow(dead_code)]
    pub language: Option<String>,
}

pub fn model_dir() -> PathBuf {
    crate::config::data_dir().join("models").join("whistle")
}

fn path_for(name: &str) -> PathBuf {
    model_dir().join(name)
}

pub fn is_downloaded() -> bool {
    let encoder = std::fs::metadata(path_for(ENCODER_NAME))
        .map(|meta| meta.len() > 100_000)
        .unwrap_or(false);
    let decoder = std::fs::metadata(path_for(DECODER_NAME))
        .map(|meta| meta.len() > 100_000)
        .unwrap_or(false);
    let encoder_data = std::fs::metadata(path_for(ENCODER_DATA_NAME))
        .map(|meta| meta.len() >= MODEL_MIN_BYTES)
        .unwrap_or(false);
    let decoder_data = std::fs::metadata(path_for(DECODER_DATA_NAME))
        .map(|meta| meta.len() >= MODEL_MIN_BYTES)
        .unwrap_or(false);
    let pack = std::fs::metadata(path_for(PACK_NAME))
        .map(|meta| meta.len() >= PACK_MIN_BYTES)
        .unwrap_or(false);
    encoder && decoder && encoder_data && decoder_data && pack
}

pub async fn ensure_downloaded() -> Result<()> {
    let _guard = DOWNLOAD_LOCK.lock().await;
    if is_downloaded() {
        return Ok(());
    }

    let dir = model_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("creating Whistle download client")?;

    for (name, minimum) in [
        (ENCODER_NAME, 100_000u64),
        (ENCODER_DATA_NAME, MODEL_MIN_BYTES),
        (DECODER_NAME, 100_000),
        (DECODER_DATA_NAME, MODEL_MIN_BYTES),
        (PACK_NAME, PACK_MIN_BYTES),
    ] {
        let url = if name == PACK_NAME {
            format!("{MODEL_REPO}/{PACK_NAME}")
        } else {
            format!("{MODEL_REPO}/onnx/{name}")
        };
        download(&client, &url, &path_for(name), minimum).await?;
    }

    if !is_downloaded() {
        bail!("Whistle model download did not produce complete files");
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

pub struct Whistle {
    encoder: Session,
    decoder: Session,
    pack: Pack,
    pub tokenizer: Tokenizer,
    engram: EngramCache,
}

impl Whistle {
    pub fn load(dir: &Path) -> Result<Self> {
        let pack = Pack::load(&dir.join(PACK_NAME))?;
        let encoder = Session::builder()
            .map_err(|error| anyhow::anyhow!("creating Whistle encoder session: {error}"))?
            .with_intra_threads(2)
            .map_err(|error| anyhow::anyhow!("configuring Whistle encoder: {error}"))?
            .commit_from_file(dir.join(ENCODER_NAME))
            .with_context(|| format!("loading {}", dir.join(ENCODER_NAME).display()))?;
        let decoder = Session::builder()
            .map_err(|error| anyhow::anyhow!("creating Whistle decoder session: {error}"))?
            .with_intra_threads(2)
            .map_err(|error| anyhow::anyhow!("configuring Whistle decoder: {error}"))?
            .commit_from_file(dir.join(DECODER_NAME))
            .with_context(|| format!("loading {}", dir.join(DECODER_NAME).display()))?;
        let tokenizer = Tokenizer {
            pieces: pack.manifest.tok.pieces.clone(),
            types: pack.manifest.tok.types.clone(),
        };
        Ok(Self {
            encoder,
            decoder,
            pack,
            tokenizer,
            engram: EngramCache::default(),
        })
    }

    pub fn transcribe(&mut self, pcm: &[f32], language: Option<&str>) -> Result<Transcription> {
        if pcm.is_empty() {
            return Ok(Transcription {
                text: String::new(),
                language: None,
            });
        }

        let (mel, frames) = mel_features(pcm, &self.pack.melfb);
        if frames < 4 {
            return Ok(Transcription {
                text: String::new(),
                language: None,
            });
        }

        let (cross_k, cross_v, cross_k_shape, cross_v_shape) = {
            let mel_tensor = Tensor::from_array(([frames, MEL_BINS], mel))?;
            let outputs = self.encoder.run(ort::inputs!["mel" => mel_tensor])?;
            let (shape_k, data_k) = outputs["cross_k"].try_extract_tensor::<f32>()?;
            let (shape_v, data_v) = outputs["cross_v"].try_extract_tensor::<f32>()?;
            (
                data_k.to_vec(),
                data_v.to_vec(),
                shape_usize(shape_k),
                shape_usize(shape_v),
            )
        };

        let mut prev = vec![0.0f32; 8 * 2 * 608];
        let mut past_k: Vec<f32> = Vec::new();
        let mut past_v: Vec<f32> = Vec::new();
        let mut past_k_shape = vec![8usize, 2, 0, 48];
        let mut past_v_shape = vec![8usize, 2, 0, 64];

        let mut tokens = vec![BOS];
        let mut ids = Vec::new();
        let mut language_index = language.and_then(|code| {
            WHISTLE_LANGUAGES
                .iter()
                .position(|candidate| *candidate == code)
        });
        let max_tokens = 320usize.min(16 + ((frames as f64 / 100.0).ceil() as usize) * 12);

        for step in 0usize..1024 {
            let token = *tokens.get(step).unwrap_or(&BOS);
            let engram_features = {
                let pack = &self.pack;
                self.engram.feat(pack, &tokens, step)
            };

            let (logits, next_prev, next_past_k, next_past_v, next_k_shape, next_v_shape) = {
                let token_tensor = Tensor::from_array(([1usize], vec![token]))?;
                let pos_tensor = Tensor::from_array(([1usize], vec![step as i64]))?;
                let engram_tensor = Tensor::from_array(([2usize, 4, 512], engram_features))?;
                let prev_ref = TensorRef::from_array_view(([8usize, 2, 608], prev.as_slice()))?;
                let past_k_ref =
                    TensorRef::from_array_view((past_k_shape.clone(), past_k.as_slice()))?;
                let past_v_ref =
                    TensorRef::from_array_view((past_v_shape.clone(), past_v.as_slice()))?;
                let cross_k_ref =
                    TensorRef::from_array_view((cross_k_shape.clone(), cross_k.as_slice()))?;
                let cross_v_ref =
                    TensorRef::from_array_view((cross_v_shape.clone(), cross_v.as_slice()))?;

                let outputs = self.decoder.run(ort::inputs![
                    "token" => token_tensor,
                    "pos" => pos_tensor,
                    "efeat" => engram_tensor,
                    "prev" => prev_ref,
                    "past_k" => past_k_ref,
                    "past_v" => past_v_ref,
                    "cross_k" => cross_k_ref,
                    "cross_v" => cross_v_ref,
                ])?;

                let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
                let (_, next_prev) = outputs["prev_out"].try_extract_tensor::<f32>()?;
                let (next_k_shape, next_past_k) =
                    outputs["present_k"].try_extract_tensor::<f32>()?;
                let (next_v_shape, next_past_v) =
                    outputs["present_v"].try_extract_tensor::<f32>()?;
                (
                    logits.to_vec(),
                    next_prev.to_vec(),
                    next_past_k.to_vec(),
                    next_past_v.to_vec(),
                    shape_usize(next_k_shape),
                    shape_usize(next_v_shape),
                )
            };

            prev = next_prev;
            past_k = next_past_k;
            past_v = next_past_v;
            past_k_shape = next_k_shape;
            past_v_shape = next_v_shape;

            if step == 0 {
                if language_index.is_none() {
                    let mut best = 0usize;
                    for i in 1..WHISTLE_LANGUAGES.len() {
                        if logits[LANG0 as usize + i] > logits[LANG0 as usize + best] {
                            best = i;
                        }
                    }
                    language_index = Some(best);
                }
                tokens.push(LANG0 + language_index.unwrap_or(0) as i64);
                continue;
            }

            let mut best = 0usize;
            for i in 1..TEXT_VOCAB {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            let next_token = best as i64;
            if next_token == EOS || ids.len() >= max_tokens {
                break;
            }
            ids.push(next_token);
            tokens.push(next_token);
        }

        let text = self.tokenizer.decode(&ids);
        Ok(Transcription {
            text: text.trim().to_string(),
            language: language_index.map(|index| WHISTLE_LANGUAGES[index].to_string()),
        })
    }
}

impl Pack {
    fn load(path: &Path) -> Result<Self> {
        let buf = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if buf.len() < 8 || &buf[0..4] != b"WPK1" {
            bail!(
                "{} is not a Whistle pack (missing WPK1 magic)",
                path.display()
            );
        }
        let manifest_len =
            u32::from_le_bytes(buf[4..8].try_into().expect("manifest length")) as usize;
        if buf.len() < 8 + manifest_len {
            bail!("{} has a truncated manifest", path.display());
        }
        let manifest: Manifest = serde_json::from_slice(&buf[8..8 + manifest_len])
            .context("parsing Whistle pack manifest")?;
        let mut base = 8 + manifest_len;
        while !base.is_multiple_of(16) {
            base += 1;
        }
        if manifest.melfb.shape != [FFT_BINS, MEL_BINS] {
            bail!(
                "unexpected Whistle mel filterbank shape: {:?}",
                manifest.melfb.shape
            );
        }
        let count = FFT_BINS * MEL_BINS;
        let start = base + manifest.melfb.src;
        let end = start
            .checked_add(count * 4)
            .filter(|end| *end <= buf.len())
            .ok_or_else(|| anyhow::anyhow!("Whistle mel filterbank is out of range"))?;
        let melfb = buf[start..end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();

        Ok(Self {
            buf,
            base,
            manifest,
            melfb,
        })
    }

    fn engram_row(&self, segment: usize, row: usize) -> Result<Vec<f32>> {
        let segment = self
            .manifest
            .engram
            .get(segment)
            .ok_or_else(|| anyhow::anyhow!("missing Whistle engram segment {segment}"))?;
        if row >= segment.out {
            bail!("Whistle engram row {row} is out of range");
        }

        let group = segment.group;
        let bits = segment.bits as usize;
        if bits != 2 && bits != 4 {
            bail!("unsupported Whistle engram bit width {bits}");
        }
        let input_padded = segment.input.div_ceil(group) * group;
        let groups = input_padded / group;
        let row_bytes = input_padded * bits / 8;
        let packed_len = segment.out * row_bytes;
        let src = self.base + segment.src;
        let packed_end = src
            .checked_add(packed_len)
            .filter(|end| *end <= self.buf.len())
            .ok_or_else(|| anyhow::anyhow!("Whistle engram data is out of range"))?;
        let packed = &self.buf[src..packed_end];
        let norm_offset = packed_end;

        let codebook = self
            .manifest
            .codebooks
            .get(&segment.bits.to_string())
            .ok_or_else(|| anyhow::anyhow!("missing Whistle {bits}-bit codebook"))?;

        let mut values = vec![0.0f64; input_padded];
        let row_start = row * row_bytes;
        if bits == 2 {
            for byte in 0..row_bytes {
                let value = packed[row_start + byte];
                let base = byte * 4;
                values[base] = codebook[(value & 3) as usize] as f64;
                values[base + 1] = codebook[((value >> 2) & 3) as usize] as f64;
                values[base + 2] = codebook[((value >> 4) & 3) as usize] as f64;
                values[base + 3] = codebook[(value >> 6) as usize] as f64;
            }
        } else {
            for byte in 0..row_bytes {
                let value = packed[row_start + byte];
                let base = byte * 2;
                values[base] = codebook[(value & 15) as usize] as f64;
                values[base + 1] = codebook[(value >> 4) as usize] as f64;
            }
        }

        let scale = 1.0 / (group as f64).sqrt();
        for group_index in 0..groups {
            let norm_pos = norm_offset + (row * groups + group_index) * 2;
            let norm_bytes: [u8; 2] = self
                .buf
                .get(norm_pos..norm_pos + 2)
                .ok_or_else(|| anyhow::anyhow!("Whistle engram norm is out of range"))?
                .try_into()
                .expect("two bytes");
            let norm = f16_to_f32(u16::from_le_bytes(norm_bytes)) as f64 * scale;
            let start = group_index * group;
            for value in &mut values[start..start + group] {
                *value *= norm;
            }
            fwht(&mut values, start, group);
        }

        Ok(values[..segment.input]
            .iter()
            .map(|value| *value as f32)
            .collect())
    }
}

impl EngramCache {
    fn feat(&mut self, pack: &Pack, tokens: &[i64], step: usize) -> Vec<f32> {
        if self.rows.is_empty() {
            self.rows = vec![HashMap::new(), HashMap::new()];
        }

        let mut features = vec![0.0f32; 2 * 4 * 512];
        let mut indices = [0usize; 4];

        for tap in 0..4usize {
            let position = match step.checked_sub(3 * tap) {
                Some(position) => position,
                None => continue,
            };
            engram_hash(tokens, position, &mut indices);
            for (table, &table_index) in indices.iter().enumerate() {
                if position < if table < 2 { 1 } else { 2 } {
                    continue;
                }
                for segment in 0..2usize {
                    let key = table * SLOTS + table_index;
                    let row = self.rows[segment]
                        .entry(key)
                        .or_insert_with(|| {
                            pack.engram_row(segment, key)
                                .map(|row| Arc::from(row.into_boxed_slice()))
                                .unwrap_or_else(|_| Arc::from(vec![0.0f32; 128].into_boxed_slice()))
                        })
                        .clone();
                    let offset = (segment * 4 + tap) * 512 + table * 128;
                    features[offset..offset + 128].copy_from_slice(&row);
                }
            }
        }

        let cached: usize = self.rows.iter().map(HashMap::len).sum();
        if cached > MAX_ENGRAM_CACHE_ROWS {
            for rows in &mut self.rows {
                rows.clear();
            }
        }

        features
    }
}

pub struct Tokenizer {
    pieces: Vec<String>,
    types: Vec<i64>,
}

impl Tokenizer {
    pub fn decode(&self, ids: &[i64]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            let index = *id as usize;
            if index >= self.pieces.len() || index >= self.types.len() {
                continue;
            }
            let piece = &self.pieces[index];
            match self.types[index] {
                4 => {
                    if let Some(hex) = piece.get(3..5) {
                        if let Ok(value) = u8::from_str_radix(hex, 16) {
                            bytes.push(value);
                        }
                    }
                }
                0 | 3 => {
                    for byte in piece.replace('▁', " ").bytes() {
                        bytes.push(byte);
                    }
                }
                _ => {}
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

fn shape_usize(shape: &ort::value::Shape) -> Vec<usize> {
    shape.as_ref().iter().map(|dim| *dim as usize).collect()
}

fn engram_hash(tokens: &[i64], position: usize, out: &mut [usize; 4]) {
    let mut index = 0usize;
    for order_index in 0..2usize {
        let order = order_index + 2;
        for hash in 0..2usize {
            let mut acc = SEED.wrapping_mul((order_index * 2 + hash + 1) as u32);
            for offset in 0..order {
                let token = if position >= offset {
                    tokens.get(position - offset).copied().unwrap_or(0)
                } else {
                    0
                };
                acc = (acc ^ (token as u32)).wrapping_mul(PRIME);
            }
            acc ^= acc >> 15;
            out[index] = (acc as usize) % SLOTS;
            index += 1;
        }
    }
}

fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let value = mantissa as f32 * 5.960_464_5e-8;
            if sign != 0 {
                (-value).to_bits()
            } else {
                value.to_bits()
            }
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

fn fwht(values: &mut [f64], offset: usize, length: usize) {
    let mut half = 1usize;
    while half * 4 <= length {
        let mut block = 0usize;
        while block < length {
            let base = offset + block;
            let mut index = 0usize;
            while index < half {
                let a0 = values[base + index];
                let a1 = values[base + index + half];
                let a2 = values[base + index + 2 * half];
                let a3 = values[base + index + 3 * half];
                let s0 = a0 + a1;
                let d0 = a0 - a1;
                let s1 = a2 + a3;
                let d1 = a2 - a3;
                values[base + index] = s0 + s1;
                values[base + index + half] = d0 + d1;
                values[base + index + 2 * half] = s0 - s1;
                values[base + index + 3 * half] = d0 - d1;
                index += 1;
            }
            block += 4 * half;
        }
        half <<= 2;
    }
    if half < length {
        let mut block = 0usize;
        while block < length {
            let base = offset + block;
            for index in 0..half {
                let a = values[base + index];
                let b = values[base + index + half];
                values[base + index] = a + b;
                values[base + index + half] = a - b;
            }
            block += 2 * half;
        }
    }
}

fn mel_features(pcm: &[f32], filterbank: &[f32]) -> (Vec<f32>, usize) {
    let sample_count = pcm.len();
    if sample_count < WIN {
        return (Vec::new(), 0);
    }

    let block_count = sample_count / 320;
    let mut gain = 1.0f64;
    if block_count > 0 {
        let mut rms = Vec::with_capacity(block_count);
        for block in 0..block_count {
            let start = block * 320;
            let end = (start + 320).min(sample_count);
            let sum: f64 = pcm[start..end]
                .iter()
                .map(|sample| (*sample as f64) * (*sample as f64))
                .sum();
            rms.push((sum / (end - start) as f64).sqrt());
        }
        rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let percentile = rms[((block_count - 1) as f64 * 0.99) as usize];
        if percentile > 1e-6 {
            gain = 0.1 / percentile;
        }
    }

    let frames = sample_count / HOP;
    let mut log_mel = vec![0.0f64; frames * MEL_BINS];
    let mut real = [0.0f64; N_FFT];
    let mut imag = [0.0f64; N_FFT];
    let mut power = [0.0f64; FFT_BINS];

    let hann: Vec<f64> = (0..WIN)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos())
        .collect();
    let (bitrev, cos, sin) = fft_tables();

    for frame in 0..frames {
        real.fill(0.0);
        imag.fill(0.0);
        let start = frame as isize * HOP as isize - (WIN as isize / 2);
        for index in 0..WIN {
            let source = start + index as isize;
            if source >= 0 && source < sample_count as isize {
                real[index] = pcm[source as usize] as f64 * gain * hann[index];
            }
        }
        fft(&mut real, &mut imag, &bitrev, &cos, &sin);
        for bin in 0..FFT_BINS {
            power[bin] = real[bin] * real[bin] + imag[bin] * imag[bin];
        }
        for mel in 0..MEL_BINS {
            let mut sum = 0.0f64;
            for bin in 0..FFT_BINS {
                sum += power[bin] * filterbank[bin * MEL_BINS + mel] as f64;
            }
            log_mel[frame * MEL_BINS + mel] = (sum + 2f64.powi(-24)).ln();
        }
    }

    let mut mel = vec![0.0f32; frames * MEL_BINS];
    for band in 0..MEL_BINS {
        let mut mean = 0.0f64;
        for frame in 0..frames {
            mean += log_mel[frame * MEL_BINS + band];
        }
        mean /= frames as f64;
        let mut variance = 0.0f64;
        for frame in 0..frames {
            let delta = log_mel[frame * MEL_BINS + band] - mean;
            variance += delta * delta;
        }
        let stddev = if frames > 1 {
            (variance / (frames - 1) as f64).sqrt()
        } else {
            0.0
        };
        for frame in 0..frames {
            mel[frame * MEL_BINS + band] =
                ((log_mel[frame * MEL_BINS + band] - mean) / (stddev + 1e-5)) as f32;
        }
    }

    (mel, frames)
}

fn fft_tables() -> ([usize; N_FFT], [f64; N_FFT / 2], [f64; N_FFT / 2]) {
    let mut bitrev = [0usize; N_FFT];
    for (index, slot) in bitrev.iter_mut().enumerate() {
        let mut reversed = 0usize;
        let mut value = index;
        for _ in 0..9 {
            reversed = (reversed << 1) | (value & 1);
            value >>= 1;
        }
        *slot = reversed;
    }
    let mut cos = [0.0f64; N_FFT / 2];
    let mut sin = [0.0f64; N_FFT / 2];
    for index in 0..N_FFT / 2 {
        let angle = -2.0 * std::f64::consts::PI * index as f64 / N_FFT as f64;
        cos[index] = angle.cos();
        sin[index] = angle.sin();
    }
    (bitrev, cos, sin)
}

fn fft(
    real: &mut [f64; N_FFT],
    imag: &mut [f64; N_FFT],
    bitrev: &[usize; N_FFT],
    cos: &[f64; N_FFT / 2],
    sin: &[f64; N_FFT / 2],
) {
    for (index, &reversed) in bitrev.iter().enumerate() {
        if reversed > index {
            real.swap(index, reversed);
            imag.swap(index, reversed);
        }
    }

    let mut size = 2usize;
    while size <= N_FFT {
        let half = size / 2;
        let step = N_FFT / size;
        let mut block = 0usize;
        while block < N_FFT {
            let mut index = 0usize;
            let mut table = 0usize;
            while index < half {
                let a = block + index;
                let b = a + half;
                let tr = real[b] * cos[table] - imag[b] * sin[table];
                let ti = real[b] * sin[table] + imag[b] * cos[table];
                real[b] = real[a] - tr;
                imag[b] = imag[a] - ti;
                real[a] += tr;
                imag[a] += ti;
                index += 1;
                table += step;
            }
            block += size;
        }
        size <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{mel_features, Tokenizer, Whistle};
    use std::path::Path;

    #[test]
    #[ignore = "downloads the Whistle ONNX model (~160 MB)"]
    fn downloads_model_and_loads() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(super::ensure_downloaded())
            .expect("download");
        assert!(super::is_downloaded());
        let _ = Whistle::load(&super::model_dir()).expect("load downloaded model");
    }

    #[test]
    #[ignore = "requires the Whistle ONNX files in /tmp/opencode/whistle-model"]
    fn transcribes_test_wav() {
        let dir = Path::new("/tmp/opencode/whistle-model");
        let mut whistle = Whistle::load(dir).expect("load Whistle");
        let mut reader = hound::WavReader::open("/tmp/opencode/whistle-test.wav").expect("wav");
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|sample| sample.expect("sample") as f32 / 32768.0)
            .collect();
        let result = whistle
            .transcribe(&samples, Some("en"))
            .expect("transcribe");
        println!("text={:?} language={:?}", result.text, result.language);
        assert!(!result.text.is_empty());
        assert!(
            result.text.to_lowercase().contains("whistle")
                || result.text.to_lowercase().contains("test"),
            "unexpected transcript: {}",
            result.text
        );
    }

    #[test]
    fn tokenizer_decode_handles_byte_pieces() {
        let tokenizer = Tokenizer {
            pieces: vec!["▁he".into(), "llo".into(), "<0x21>".into()],
            types: vec![0, 0, 4],
        };
        assert_eq!(tokenizer.decode(&[0, 1, 2]), " hello!");
    }

    #[test]
    fn mel_features_are_normalized() {
        let filterbank = vec![0.001f32; 257 * 80];
        let pcm = vec![0.0f32; 16000];
        let (mel, frames) = mel_features(&pcm, &filterbank);
        assert_eq!(frames, 100);
        assert_eq!(mel.len(), frames * 80);
        assert!(mel.iter().all(|value| value.abs() < 1e-3));
    }
}
