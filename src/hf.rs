use std::collections::HashSet;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

/// A downloadable GGUF file in a Hugging Face repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfQuant {
    /// Quantization tag accepted by Ollama, e.g. `Q4_K_M`.
    pub tag: String,
    /// Repository-relative file name, used as a fallback tag and in errors.
    pub file: String,
    /// File size in bytes when the Hugging Face API reports one.
    pub size: Option<u64>,
}

/// Normalize user input into `(owner/repository, optional quantization)`.
///
/// Accepts `owner/repo`, `hf.co/owner/repo`, `https://huggingface.co/owner/repo`,
/// and an optional `:Q4_K_M` suffix.
pub fn normalize_repo(input: &str) -> Result<(String, Option<String>)> {
    let mut text = input.trim();
    for prefix in [
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://hf.co/",
        "http://hf.co/",
        "huggingface.co/",
        "hf.co/",
    ] {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest;
            break;
        }
    }
    let text = text.trim().trim_end_matches('/');
    if text.is_empty() {
        bail!("enter a Hugging Face repository");
    }

    let (repo, tag) = match text.rsplit_once(':') {
        Some((repo, tag)) => {
            let tag = tag.trim();
            if tag.is_empty() {
                (repo, None)
            } else {
                if !tag
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
                {
                    bail!("quantization tags may only contain letters, numbers, '-', '_' and '.'");
                }
                (repo, Some(tag.to_ascii_uppercase()))
            }
        }
        None => (text, None),
    };

    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|part| part.is_empty()) {
        bail!("expected owner/repository, e.g. AtomicChat/LFM3-1.7B-Uncensored-GGUF");
    }
    for part in &parts {
        if *part == "." || *part == ".." {
            bail!("repository names may not be '.' or '..'");
        }
        if !part
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        {
            bail!("repository names may only contain letters, numbers, '-', '_' and '.'");
        }
    }

    Ok((repo.to_string(), tag))
}

/// The short Ollama model name shown in the model list. The quantization is
/// still used for the pull reference, but it is not part of the local name.
pub fn short_name(repo: &str) -> Result<String> {
    let name = repo.rsplit('/').next().unwrap_or(repo).trim();
    if name.is_empty() {
        bail!("repository name is empty");
    }
    Ok(name.to_string())
}

/// The reference Ollama expects when pulling directly from Hugging Face.
pub fn pull_reference(repo: &str, tag: &str) -> String {
    format!("hf.co/{repo}:{tag}")
}

/// Fetch the available GGUF quantizations for a repository.
pub async fn fetch_quantizations(repo: &str) -> Result<Vec<HfQuant>> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(25))
        .user_agent(concat!(
            "ollama-gui/",
            env!("CARGO_PKG_VERSION"),
            " (+https://github.com/tibssy/Ollama-GUI-rs)"
        ))
        .build()
        .context("could not create the Hugging Face client")?;
    let response = client
        .get(&url)
        .send()
        .await
        .context("could not reach Hugging Face")?;
    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        bail!("this Hugging Face repository is gated or private");
    }
    if status == reqwest::StatusCode::NOT_FOUND {
        bail!("Hugging Face repository not found");
    }
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let detail = body.trim();
        if detail.is_empty() {
            bail!("Hugging Face returned HTTP {status}");
        }
        bail!("Hugging Face returned HTTP {status}: {detail}");
    }

    let entries: Vec<TreeEntry> = response
        .json()
        .await
        .context("unexpected Hugging Face tree payload")?;
    Ok(quantizations_from_entries(entries))
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    size: Option<u64>,
}

fn quantizations_from_entries(entries: Vec<TreeEntry>) -> Vec<HfQuant> {
    let mut quants = Vec::new();
    let mut seen = HashSet::new();

    for entry in entries {
        if !entry.kind.is_empty() && entry.kind != "file" {
            continue;
        }
        let lower = entry.path.to_ascii_lowercase();
        if !lower.ends_with(".gguf") {
            continue;
        }
        // Projector files are not standalone Ollama models.
        if lower.contains("mmproj") || lower.contains("imatrix") {
            continue;
        }

        let tag = parse_quant(&entry.path).unwrap_or_else(|| file_stem(&entry.path));
        if tag.is_empty() || !seen.insert(tag.clone()) {
            continue;
        }
        quants.push(HfQuant {
            tag,
            file: entry.path,
            size: entry.size.filter(|size| *size > 0),
        });
    }

    quants.sort_by_key(|quant| (quant_rank(&quant.tag), quant.tag.clone()));
    quants
}

/// Extract a known quantization tag from a GGUF file name.
pub fn parse_quant(path: &str) -> Option<String> {
    let stem = file_stem(path);
    let upper = stem.to_ascii_uppercase();
    let tokens: Vec<&str> = upper
        .split(['-', '_', '.', ' '])
        .filter(|token| !token.is_empty())
        .collect();

    for length in (1..=3.min(tokens.len())).rev() {
        let candidate = tokens[tokens.len() - length..].join("_");
        if is_known_quant(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn file_stem(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem = if name.to_ascii_lowercase().ends_with(".gguf") {
        &name[..name.len() - 5]
    } else {
        name
    };
    strip_split_suffix(stem).to_string()
}

/// Split GGUF files are named `...-00001-of-00002.gguf`. Remove that suffix so
/// every part maps to the same quantization.
fn strip_split_suffix(stem: &str) -> &str {
    if let Some((left, right)) = stem.rsplit_once("-of-") {
        if !right.is_empty() && right.chars().all(|c| c.is_ascii_digit()) {
            if let Some((base, index)) = left.rsplit_once('-') {
                if !index.is_empty() && index.chars().all(|c| c.is_ascii_digit()) {
                    return base;
                }
            }
        }
    }
    stem
}

fn is_known_quant(tag: &str) -> bool {
    matches!(
        tag,
        "Q2_K"
            | "Q2_K_S"
            | "Q2_K_L"
            | "Q3_K_S"
            | "Q3_K_M"
            | "Q3_K_L"
            | "Q4_0"
            | "Q4_1"
            | "Q4_K_S"
            | "Q4_K_M"
            | "Q5_0"
            | "Q5_1"
            | "Q5_K_S"
            | "Q5_K_M"
            | "Q6_K"
            | "Q8_0"
            | "F16"
            | "FP16"
            | "BF16"
            | "F32"
            | "IQ1_S"
            | "IQ1_M"
            | "IQ2_XXS"
            | "IQ2_XS"
            | "IQ2_S"
            | "IQ2_M"
            | "IQ3_XXS"
            | "IQ3_XS"
            | "IQ3_S"
            | "IQ3_M"
            | "IQ4_NL"
            | "IQ4_XS"
    )
}

fn quant_rank(tag: &str) -> usize {
    match tag {
        "Q4_K_M" => 0,
        "Q5_K_M" => 1,
        "Q4_K_S" => 2,
        "Q5_K_S" => 3,
        "Q6_K" => 4,
        "Q8_0" => 5,
        "Q3_K_M" => 6,
        "Q2_K" => 7,
        _ => 10,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_repo, parse_quant, pull_reference, quantizations_from_entries, short_name,
        TreeEntry,
    };

    #[test]
    fn normalizes_repository_inputs() {
        assert_eq!(
            normalize_repo("AtomicChat/LFM3-1.7B-Uncensored-GGUF").unwrap(),
            ("AtomicChat/LFM3-1.7B-Uncensored-GGUF".to_string(), None)
        );
        assert_eq!(
            normalize_repo("hf.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF:q4_k_m").unwrap(),
            (
                "AtomicChat/LFM3-1.7B-Uncensored-GGUF".to_string(),
                Some("Q4_K_M".to_string())
            )
        );
        assert_eq!(
            normalize_repo("https://huggingface.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF/").unwrap(),
            ("AtomicChat/LFM3-1.7B-Uncensored-GGUF".to_string(), None)
        );
        assert!(normalize_repo("").is_err());
        assert!(normalize_repo("not-a-repo").is_err());
        assert!(normalize_repo("owner/repo/extra").is_err());
        assert!(normalize_repo("../repo").is_err());
        assert!(normalize_repo("owner/..").is_err());
    }

    #[test]
    fn builds_short_names_and_pull_references() {
        let repo = "AtomicChat/LFM3-1.7B-Uncensored-GGUF";
        assert_eq!(short_name(repo).unwrap(), "LFM3-1.7B-Uncensored-GGUF");
        assert_eq!(
            pull_reference(repo, "Q4_K_M"),
            "hf.co/AtomicChat/LFM3-1.7B-Uncensored-GGUF:Q4_K_M"
        );
    }

    #[test]
    fn parses_quantization_tags() {
        assert_eq!(
            parse_quant("LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(parse_quant("model-Q6_K.gguf").as_deref(), Some("Q6_K"));
        assert_eq!(parse_quant("model-IQ4_XS.gguf").as_deref(), Some("IQ4_XS"));
        assert_eq!(parse_quant("model-BF16.gguf").as_deref(), Some("BF16"));
        assert_eq!(
            parse_quant("model-Q4_K_M-00001-of-00002.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(parse_quant("model.gguf"), None);
    }

    #[test]
    fn skips_projectors_and_deduplicates_quants() {
        let entries = vec![
            TreeEntry {
                kind: "file".to_string(),
                path: "model-Q4_K_M.gguf".to_string(),
                size: Some(1_000),
            },
            TreeEntry {
                kind: "file".to_string(),
                path: "model-Q4_K_M-00001-of-00002.gguf".to_string(),
                size: Some(1_000),
            },
            TreeEntry {
                kind: "file".to_string(),
                path: "mmproj-model-f16.gguf".to_string(),
                size: Some(500),
            },
            TreeEntry {
                kind: "directory".to_string(),
                path: "sub".to_string(),
                size: None,
            },
        ];
        let quants = quantizations_from_entries(entries);
        assert_eq!(quants.len(), 1);
        assert_eq!(quants[0].tag, "Q4_K_M");
    }

    #[test]
    fn orders_q4_k_m_first() {
        let entries = vec![
            TreeEntry {
                kind: "file".to_string(),
                path: "model-Q8_0.gguf".to_string(),
                size: Some(1_000),
            },
            TreeEntry {
                kind: "file".to_string(),
                path: "model-Q4_K_M.gguf".to_string(),
                size: Some(1_000),
            },
        ];
        let quants = quantizations_from_entries(entries);
        assert_eq!(quants[0].tag, "Q4_K_M");
        assert_eq!(quants[1].tag, "Q8_0");
    }
}
