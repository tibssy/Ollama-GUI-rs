use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use crate::config::{self, Settings, ToolPermission, ToolPermissions};

pub const REGISTRY_BASE: &str = "https://raw.githubusercontent.com/tibssy/Ollama-GUI-Tools/main";
const MAX_REGISTRY_FILE_BYTES: usize = 1024 * 1024;
const REGISTRY_TIMEOUT_SECS: u64 = 20;
const MAX_FILE_PARAMS: usize = 8;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolRequires {
    #[serde(default)]
    pub commands: Vec<String>,
    #[serde(default)]
    pub python: Vec<String>,
    #[serde(default)]
    pub arch: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolSandbox {
    #[serde(default)]
    pub network: bool,
    #[serde(default)]
    pub wayland: bool,
    #[serde(default)]
    pub dbus: bool,
    #[serde(default)]
    pub runtime_dir: bool,
    #[serde(default)]
    pub read_paths: Vec<String>,
    #[serde(default)]
    pub write_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolManifest {
    pub name: String,
    pub description: String,
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default = "default_category")]
    pub category: String,
    #[serde(default = "default_runtime")]
    pub runtime: String,
    #[serde(default)]
    pub requires: ToolRequires,
    #[serde(default)]
    pub sandbox: ToolSandbox,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_max_output")]
    pub max_output_bytes: usize,
    #[serde(default)]
    pub file_params: Vec<String>,
    #[serde(default)]
    pub dangerous_actions: Vec<String>,
    #[serde(default = "default_parameters")]
    pub parameters: Value,
    #[serde(default = "default_permission")]
    pub default_permission: String,
    #[serde(default = "default_cloud_permission")]
    pub default_cloud_permission: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolIndex {
    #[serde(default)]
    pub schema: u32,
    #[serde(default)]
    pub tools: Vec<ToolIndexEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolIndexEntry {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default = "default_category")]
    pub category: String,
    #[serde(default = "default_runtime")]
    pub runtime: String,
    #[serde(default)]
    pub requires: ToolRequires,
    #[serde(default = "default_permission")]
    pub default_permission: String,
    #[serde(default = "default_cloud_permission")]
    pub default_cloud_permission: String,
    #[serde(default)]
    pub files: BTreeMap<String, ToolFile>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolFile {
    pub sha256: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Clone)]
pub struct InstalledTool {
    pub manifest: ToolManifest,
    pub dir: PathBuf,
}

impl InstalledTool {
    #[allow(dead_code)]
    pub fn id(&self) -> &str {
        &self.manifest.name
    }

    pub fn run_path(&self) -> PathBuf {
        self.dir.join("run")
    }

    #[allow(dead_code)]
    pub fn definition(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.manifest.name,
                "description": self.manifest.description,
                "parameters": self.manifest.parameters,
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct ToolPackage {
    pub entry: ToolIndexEntry,
    pub manifest: ToolManifest,
    pub manifest_bytes: Vec<u8>,
    pub run_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ToolRun {
    pub success: bool,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct SandboxPlan {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub env: Vec<(OsString, OsString)>,
    pub input: Value,
    pub timeout: Duration,
    pub max_output_bytes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct RequirementStatus {
    pub missing_commands: Vec<String>,
    pub missing_python: Vec<String>,
    pub unsupported_arch: bool,
}

impl RequirementStatus {
    pub fn is_ok(&self) -> bool {
        self.missing_commands.is_empty() && self.missing_python.is_empty() && !self.unsupported_arch
    }

    pub fn summary(&self) -> Option<String> {
        let mut parts = Vec::new();
        if !self.missing_commands.is_empty() {
            parts.push(format!("missing: {}", self.missing_commands.join(", ")));
        }
        if !self.missing_python.is_empty() {
            parts.push(format!(
                "missing Python: {}",
                self.missing_python.join(", ")
            ));
        }
        if self.unsupported_arch {
            parts.push(format!(
                "unsupported architecture ({})",
                std::env::consts::ARCH
            ));
        }
        (!parts.is_empty()).then(|| parts.join(" · "))
    }
}

fn default_version() -> u32 {
    1
}

fn default_category() -> String {
    "other".to_string()
}

fn default_runtime() -> String {
    "python3".to_string()
}

fn default_timeout() -> u64 {
    30
}

fn default_max_output() -> usize {
    65_536
}

fn default_parameters() -> Value {
    json!({ "type": "object", "properties": {} })
}

fn default_permission() -> String {
    "ask".to_string()
}

fn default_cloud_permission() -> String {
    "deny".to_string()
}

fn valid_permission(value: &str) -> bool {
    matches!(value, "allow" | "ask" | "deny")
}

pub fn is_valid_id(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(first) if first.is_ascii_lowercase() => {}
        _ => return false,
    }
    (2..=64).contains(&id.len())
        && id
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit())
}

pub fn tools_dir() -> PathBuf {
    config::data_dir().join("tools")
}

pub fn list_installed() -> Vec<InstalledTool> {
    let dir = tools_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut tools = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        match load_installed(&path) {
            Ok(tool) => tools.push(tool),
            Err(err) => {
                tracing::warn!("skipping tool at {}: {err:#}", path.display());
            }
        }
    }
    tools.sort_by(|left, right| left.manifest.name.cmp(&right.manifest.name));
    tools
}

pub fn load_installed(dir: &Path) -> Result<InstalledTool> {
    let name = dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("invalid tool directory"))?
        .to_string();
    let manifest_path = dir.join("tool.json");
    let contents = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("could not read {}", manifest_path.display()))?;
    let manifest: ToolManifest =
        serde_json::from_str(&contents).context("invalid tool manifest")?;
    validate_manifest(&name, &manifest)?;
    if !dir.join("run").is_file() {
        bail!("missing run script");
    }
    Ok(InstalledTool {
        manifest,
        dir: dir.to_path_buf(),
    })
}

pub fn validate_manifest(id: &str, manifest: &ToolManifest) -> Result<()> {
    if !is_valid_id(id) {
        bail!("invalid tool id '{id}'");
    }
    if manifest.name != id {
        bail!(
            "manifest name '{}' does not match directory '{id}'",
            manifest.name
        );
    }
    if manifest.description.trim().is_empty() {
        bail!("description cannot be empty");
    }
    if !matches!(manifest.runtime.as_str(), "python3" | "binary") {
        bail!("unsupported runtime '{}'", manifest.runtime);
    }
    if !manifest.parameters.is_object()
        || manifest.parameters.get("type").and_then(Value::as_str) != Some("object")
    {
        bail!("parameters must be a JSON schema object");
    }
    if !valid_permission(&manifest.default_permission) {
        bail!(
            "invalid default_permission '{}'",
            manifest.default_permission
        );
    }
    if !valid_permission(&manifest.default_cloud_permission) {
        bail!(
            "invalid default_cloud_permission '{}'",
            manifest.default_cloud_permission
        );
    }
    if !(1..=300).contains(&manifest.timeout_secs) {
        bail!("timeout_secs must be between 1 and 300");
    }
    if !(1024..=1024 * 1024).contains(&manifest.max_output_bytes) {
        bail!("max_output_bytes must be between 1024 and 1048576");
    }

    let properties = manifest
        .parameters
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for name in &manifest.file_params {
        if !properties.contains_key(name) {
            bail!("file parameter '{name}' is not declared in parameters");
        }
    }
    if manifest.file_params.len() > MAX_FILE_PARAMS {
        bail!("too many file parameters");
    }
    for name in &manifest.requires.commands {
        if name.trim().is_empty() {
            bail!("required command names cannot be empty");
        }
    }
    for value in manifest
        .sandbox
        .read_paths
        .iter()
        .chain(manifest.sandbox.write_paths.iter())
    {
        validate_sandbox_path(value)?;
    }
    Ok(())
}

fn validate_sandbox_path(value: &str) -> Result<()> {
    if value.is_empty() || value.contains('\0') {
        bail!("sandbox paths cannot be empty or contain NUL bytes");
    }
    if !(value.starts_with('/') || value.starts_with("~/") || value == "~") {
        bail!("sandbox path '{value}' must be absolute or start with ~/");
    }
    let expanded = expand_path(value);
    if expanded
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        bail!("sandbox path '{value}' must not contain '..'");
    }
    Ok(())
}

fn validate_index_entry(entry: &ToolIndexEntry) -> Result<()> {
    if !is_valid_id(&entry.id) {
        bail!("invalid registry tool id '{}'", entry.id);
    }
    if !matches!(entry.runtime.as_str(), "python3" | "binary") {
        bail!("{}: unsupported runtime '{}'", entry.id, entry.runtime);
    }
    if !valid_permission(&entry.default_permission)
        || !valid_permission(&entry.default_cloud_permission)
    {
        bail!("{}: invalid default permission", entry.id);
    }
    for required in ["tool.json", "run"] {
        let file = entry
            .files
            .get(required)
            .ok_or_else(|| anyhow!("{}: missing file entry for {required}", entry.id))?;
        if !is_sha256_hex(&file.sha256) {
            bail!("{}: invalid sha256 for {required}", entry.id);
        }
        if file.size as usize > MAX_REGISTRY_FILE_BYTES {
            bail!("{}: {required} is too large", entry.id);
        }
    }
    if entry.files.len() != 2 {
        bail!("{}: unexpected files in index entry", entry.id);
    }
    Ok(())
}

pub fn find_command(name: &str) -> Option<PathBuf> {
    if name.contains('/') {
        let path = PathBuf::from(name);
        return path.is_file().then_some(path);
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

pub fn missing_commands(manifest: &ToolManifest) -> Vec<String> {
    manifest
        .requires
        .commands
        .iter()
        .filter(|name| find_command(name).is_none())
        .cloned()
        .collect()
}

pub fn missing_python_modules(manifest: &ToolManifest) -> Vec<String> {
    if manifest.requires.python.is_empty() {
        return Vec::new();
    }
    let Some(python) = find_command("python3") else {
        return manifest.requires.python.clone();
    };
    manifest
        .requires
        .python
        .iter()
        .filter(|module| {
            let script = format!(
                "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec({module:?}) else 1)"
            );
            std::process::Command::new(&python)
                .arg("-c")
                .arg(script)
                .env_remove("PYTHONPATH")
                .env("PYTHONNOUSERSITE", "1")
                .status()
                .map(|status| !status.success())
                .unwrap_or(true)
        })
        .cloned()
        .collect()
}

pub fn arch_supported(manifest: &ToolManifest) -> bool {
    manifest.requires.arch.is_empty()
        || manifest
            .requires
            .arch
            .iter()
            .any(|arch| arch == std::env::consts::ARCH)
}

pub fn requirement_status(manifest: &ToolManifest) -> RequirementStatus {
    RequirementStatus {
        missing_commands: missing_commands(manifest),
        missing_python: missing_python_modules(manifest),
        unsupported_arch: !arch_supported(manifest),
    }
}

pub fn permission_for(settings: &Settings, manifest: &ToolManifest, cloud: bool) -> ToolPermission {
    let defaults = ToolPermissions::from_manifest(
        &manifest.default_permission,
        &manifest.default_cloud_permission,
    );
    let permissions = settings
        .tool_permissions
        .get(&manifest.name)
        .cloned()
        .unwrap_or(defaults);
    if cloud {
        permissions.cloud
    } else {
        permissions.local
    }
}

#[allow(dead_code)]
pub fn is_dangerous_action(manifest: &ToolManifest, args: &Value) -> bool {
    let Some(action) = args.get("action").and_then(Value::as_str) else {
        return false;
    };
    manifest
        .dangerous_actions
        .iter()
        .any(|value| value == action)
}

fn registry_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(concat!("ollama-gui/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(REGISTRY_TIMEOUT_SECS))
        .build()
        .context("could not create the registry HTTP client")
}

async fn fetch_bytes(client: &reqwest::Client, url: &str) -> Result<Vec<u8>> {
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("could not reach {url}"))?;
    let status = response.status();
    if !status.is_success() {
        bail!("tool registry returned {status} for {url}");
    }
    if let Some(length) = response.content_length() {
        if length as usize > MAX_REGISTRY_FILE_BYTES {
            bail!("{url} is larger than the allowed registry file size");
        }
    }
    let mut stream = response.bytes_stream();
    let mut data = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("could not download {url}"))?;
        if data.len() + chunk.len() > MAX_REGISTRY_FILE_BYTES {
            bail!("{url} is larger than the allowed registry file size");
        }
        data.extend_from_slice(&chunk);
    }
    Ok(data)
}

pub async fn fetch_index() -> Result<ToolIndex> {
    let client = registry_client()?;
    fetch_index_with(&client).await
}

pub async fn fetch_index_with(client: &reqwest::Client) -> Result<ToolIndex> {
    let bytes = fetch_bytes(client, &format!("{REGISTRY_BASE}/index.json")).await?;
    let index: ToolIndex = serde_json::from_slice(&bytes).context("invalid tool index")?;
    if index.schema != 1 {
        bail!("unsupported tool index schema {}", index.schema);
    }
    for entry in &index.tools {
        validate_index_entry(entry)?;
    }
    Ok(index)
}

pub async fn fetch_package(entry: &ToolIndexEntry) -> Result<ToolPackage> {
    let client = registry_client()?;
    fetch_package_with(&client, entry).await
}

pub async fn fetch_package_with(
    client: &reqwest::Client,
    entry: &ToolIndexEntry,
) -> Result<ToolPackage> {
    validate_index_entry(entry)?;
    let base = format!("{REGISTRY_BASE}/tools/{}", entry.id);
    let manifest_bytes = fetch_bytes(client, &format!("{base}/tool.json")).await?;
    let run_bytes = fetch_bytes(client, &format!("{base}/run")).await?;
    verify_file(entry, "tool.json", &manifest_bytes)?;
    verify_file(entry, "run", &run_bytes)?;

    let manifest: ToolManifest =
        serde_json::from_slice(&manifest_bytes).context("invalid tool manifest")?;
    validate_manifest(&entry.id, &manifest)?;
    if manifest.name != entry.id {
        bail!("manifest name does not match the registry id");
    }
    Ok(ToolPackage {
        entry: entry.clone(),
        manifest,
        manifest_bytes,
        run_bytes,
    })
}

fn verify_file(entry: &ToolIndexEntry, name: &str, bytes: &[u8]) -> Result<()> {
    let expected = entry
        .files
        .get(name)
        .ok_or_else(|| anyhow!("{}: no hash for {name}", entry.id))?;
    if expected.size as usize != bytes.len() {
        bail!(
            "{}: size mismatch for {name} (expected {}, got {})",
            entry.id,
            expected.size,
            bytes.len()
        );
    }
    let digest = sha256_hex(bytes);
    if digest != expected.sha256.to_ascii_lowercase() {
        bail!("{}: checksum mismatch for {name}", entry.id);
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn install_package(package: &ToolPackage) -> Result<InstalledTool> {
    validate_index_entry(&package.entry)?;
    validate_manifest(&package.entry.id, &package.manifest)?;
    verify_file(&package.entry, "tool.json", &package.manifest_bytes)?;
    verify_file(&package.entry, "run", &package.run_bytes)?;

    let root = tools_dir();
    std::fs::create_dir_all(&root)
        .with_context(|| format!("could not create {}", root.display()))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let id = &package.entry.id;
    let temp = root.join(format!(".tmp-{id}-{stamp}"));
    let final_dir = root.join(id);
    if temp.exists() {
        std::fs::remove_dir_all(&temp).ok();
    }
    std::fs::create_dir_all(&temp)?;

    let result = (|| -> Result<()> {
        std::fs::write(temp.join("tool.json"), &package.manifest_bytes)?;
        std::fs::write(temp.join("run"), &package.run_bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(temp.join("run"))?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(temp.join("run"), perms)?;
        }

        let backup = root.join(format!(".old-{id}-{stamp}"));
        if final_dir.exists() {
            std::fs::rename(&final_dir, &backup)?;
        }
        match std::fs::rename(&temp, &final_dir) {
            Ok(()) => {
                let _ = std::fs::remove_dir_all(&backup);
                Ok(())
            }
            Err(err) => {
                if backup.exists() {
                    let _ = std::fs::rename(&backup, &final_dir);
                }
                Err(err.into())
            }
        }
    })();

    if result.is_err() {
        let _ = std::fs::remove_dir_all(&temp);
    }
    result?;

    Ok(InstalledTool {
        manifest: package.manifest.clone(),
        dir: final_dir,
    })
}

pub fn uninstall(id: &str) -> Result<()> {
    if !is_valid_id(id) {
        bail!("invalid tool id '{id}'");
    }
    let dir = tools_dir().join(id);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("could not delete {}", dir.display()))?;
    }
    Ok(())
}

fn expand_path(value: &str) -> PathBuf {
    if value == "~" {
        config::home_dir()
    } else if let Some(rest) = value.strip_prefix("~/") {
        config::home_dir().join(rest)
    } else {
        PathBuf::from(value)
    }
}

fn sandbox_destination(value: &str) -> PathBuf {
    if value == "~" {
        PathBuf::from("/sandbox/home")
    } else if let Some(rest) = value.strip_prefix("~/") {
        PathBuf::from("/sandbox/home").join(rest)
    } else {
        PathBuf::from(value)
    }
}

fn add_dir(args: &mut Vec<OsString>, dirs: &mut BTreeSet<PathBuf>, path: &Path) {
    let mut ancestors: Vec<&Path> = path.ancestors().collect();
    ancestors.reverse();
    for ancestor in ancestors {
        if ancestor == Path::new("/") || ancestor.as_os_str().is_empty() {
            continue;
        }
        if dirs.insert(ancestor.to_path_buf()) {
            args.push(OsString::from("--dir"));
            args.push(ancestor.as_os_str().to_os_string());
        }
    }
}

fn add_bind(
    args: &mut Vec<OsString>,
    dirs: &mut BTreeSet<PathBuf>,
    source: &Path,
    destination: &Path,
    writable: bool,
) {
    if let Some(parent) = destination.parent() {
        add_dir(args, dirs, parent);
    }
    args.push(OsString::from(if writable {
        "--bind"
    } else {
        "--ro-bind"
    }));
    args.push(source.as_os_str().to_os_string());
    args.push(destination.as_os_str().to_os_string());
}

fn add_system_path(args: &mut Vec<OsString>, path: &str) {
    let host = Path::new(path);
    if !host.exists() {
        return;
    }
    if host.is_symlink() {
        if let Ok(target) = std::fs::read_link(host) {
            args.push(OsString::from("--symlink"));
            args.push(target.as_os_str().to_os_string());
            args.push(OsString::from(path));
        }
    } else {
        args.push(OsString::from("--ro-bind"));
        args.push(OsString::from(path));
        args.push(OsString::from(path));
    }
}

fn runtime_dir_path() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
}

fn dbus_socket_path(runtime_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = runtime_dir {
        let candidate = dir.join("bus");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    let address = std::env::var("DBUS_SESSION_BUS_ADDRESS").ok()?;
    for part in address.split(';') {
        for key in part.split(',') {
            if let Some(path) = key.strip_prefix("unix:path=") {
                return Some(PathBuf::from(path));
            }
        }
    }
    None
}

pub fn build_sandbox_plan(tool: &InstalledTool, args: &Value) -> Result<SandboxPlan> {
    if find_command("bwrap").is_none() {
        bail!("bwrap is required to run tools. Install bubblewrap and try again.");
    }
    let manifest = &tool.manifest;
    let missing = requirement_status(manifest);
    if !missing.missing_commands.is_empty() {
        bail!(
            "Missing required command(s): {}",
            missing.missing_commands.join(", ")
        );
    }
    if !missing.missing_python.is_empty() {
        bail!(
            "Missing required Python module(s): {}",
            missing.missing_python.join(", ")
        );
    }
    if missing.unsupported_arch {
        bail!(
            "This tool does not support the {} architecture",
            std::env::consts::ARCH
        );
    }

    let mut command_args: Vec<OsString> = vec![
        "--die-with-parent".into(),
        "--unshare-all".into(),
        "--ro-bind".into(),
        "/usr".into(),
        "/usr".into(),
        "--ro-bind".into(),
        "/etc".into(),
        "/etc".into(),
        "--proc".into(),
        "/proc".into(),
        "--dev".into(),
        "/dev".into(),
        "--tmpfs".into(),
        "/tmp".into(),
    ];
    for path in ["/bin", "/lib", "/lib64", "/sbin", "/usr/local"] {
        add_system_path(&mut command_args, path);
    }

    let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
    for path in [
        "/sandbox",
        "/sandbox/home",
        "/sandbox/tool",
        "/sandbox/incoming",
    ] {
        add_dir(&mut command_args, &mut dirs, Path::new(path));
    }
    command_args.push("--ro-bind".into());
    command_args.push(tool.dir.as_os_str().to_os_string());
    command_args.push("/sandbox/tool".into());
    command_args.push("--chdir".into());
    command_args.push("/sandbox/tool".into());

    let mut env: Vec<(OsString, OsString)> = vec![
        ("HOME".into(), "/sandbox/home".into()),
        ("PATH".into(), "/usr/local/bin:/usr/bin:/bin".into()),
        ("LANG".into(), "C.UTF-8".into()),
        ("LC_ALL".into(), "C.UTF-8".into()),
        ("TMPDIR".into(), "/tmp".into()),
        ("PYTHONDONTWRITEBYTECODE".into(), "1".into()),
        ("PYTHONNOUSERSITE".into(), "1".into()),
        ("XDG_DATA_HOME".into(), "/sandbox/home/.local/share".into()),
        ("XDG_CONFIG_HOME".into(), "/sandbox/home/.config".into()),
        ("XDG_CACHE_HOME".into(), "/sandbox/home/.cache".into()),
    ];

    if manifest.sandbox.network {
        command_args.push("--share-net".into());
    }

    let runtime_dir = runtime_dir_path();
    let display = std::env::var("WAYLAND_DISPLAY")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "wayland-0".to_string());

    if manifest.sandbox.runtime_dir {
        let Some(runtime_dir) = runtime_dir.as_deref() else {
            bail!("XDG_RUNTIME_DIR is not set; this tool needs the desktop runtime directory");
        };
        add_bind(
            &mut command_args,
            &mut dirs,
            runtime_dir,
            Path::new("/sandbox/run"),
            true,
        );
        env.push(("XDG_RUNTIME_DIR".into(), "/sandbox/run".into()));
    }

    if manifest.sandbox.wayland {
        if manifest.sandbox.runtime_dir {
            env.push(("WAYLAND_DISPLAY".into(), display.into()));
        } else {
            let Some(runtime_dir) = runtime_dir.as_deref() else {
                bail!("XDG_RUNTIME_DIR is not set; this tool needs Wayland");
            };
            let socket = runtime_dir.join(&display);
            if !socket.exists() {
                bail!("Wayland socket not found at {}", socket.display());
            }
            add_dir(&mut command_args, &mut dirs, Path::new("/sandbox/run"));
            command_args.push("--ro-bind".into());
            command_args.push(socket.as_os_str().to_os_string());
            command_args.push("/sandbox/run/wayland-0".into());
            env.push(("XDG_RUNTIME_DIR".into(), "/sandbox/run".into()));
            env.push(("WAYLAND_DISPLAY".into(), "wayland-0".into()));
        }
    }

    if manifest.sandbox.dbus {
        let Some(runtime_dir) = runtime_dir.as_deref() else {
            bail!("XDG_RUNTIME_DIR is not set; this tool needs D-Bus");
        };
        let Some(bus) = dbus_socket_path(Some(runtime_dir)) else {
            bail!("Could not find the D-Bus session bus");
        };
        let bus_path = if manifest.sandbox.runtime_dir {
            match bus.strip_prefix(runtime_dir) {
                Ok(relative) => PathBuf::from("/sandbox/run").join(relative),
                Err(_) => {
                    command_args.push("--ro-bind".into());
                    command_args.push(bus.as_os_str().to_os_string());
                    command_args.push("/sandbox/run/bus".into());
                    PathBuf::from("/sandbox/run/bus")
                }
            }
        } else {
            add_dir(&mut command_args, &mut dirs, Path::new("/sandbox/run"));
            command_args.push("--ro-bind".into());
            command_args.push(bus.as_os_str().to_os_string());
            command_args.push("/sandbox/run/bus".into());
            PathBuf::from("/sandbox/run/bus")
        };
        env.push(("XDG_RUNTIME_DIR".into(), "/sandbox/run".into()));
        env.push((
            "DBUS_SESSION_BUS_ADDRESS".into(),
            format!("unix:path={}", bus_path.display()).into(),
        ));
    }

    if manifest.sandbox.wayland && Path::new("/dev/dri").exists() {
        command_args.push("--dev-bind".into());
        command_args.push("/dev/dri".into());
        command_args.push("/dev/dri".into());
    }

    for raw in &manifest.sandbox.read_paths {
        let source = expand_path(raw);
        if !source.exists() {
            tracing::warn!("tool {}: read path {} does not exist", manifest.name, raw);
            continue;
        }
        let destination = sandbox_destination(raw);
        add_bind(&mut command_args, &mut dirs, &source, &destination, false);
    }
    for raw in &manifest.sandbox.write_paths {
        let source = expand_path(raw);
        if !source.exists() {
            tracing::warn!("tool {}: write path {} does not exist", manifest.name, raw);
            continue;
        }
        let destination = sandbox_destination(raw);
        add_bind(&mut command_args, &mut dirs, &source, &destination, true);
    }

    let input = stage_file_params(tool, args, &mut command_args, &mut dirs)?;

    command_args.push("--".into());
    match manifest.runtime.as_str() {
        "python3" => {
            command_args.push("python3".into());
            command_args.push("/sandbox/tool/run".into());
        }
        "binary" => {
            command_args.push("/sandbox/tool/run".into());
        }
        other => bail!("unsupported runtime '{other}'"),
    }

    Ok(SandboxPlan {
        program: OsString::from("bwrap"),
        args: command_args,
        env,
        input,
        timeout: Duration::from_secs(manifest.timeout_secs.clamp(1, 300)),
        max_output_bytes: manifest.max_output_bytes,
    })
}

fn stage_file_params(
    tool: &InstalledTool,
    input: &Value,
    command_args: &mut Vec<OsString>,
    dirs: &mut BTreeSet<PathBuf>,
) -> Result<Value> {
    let mut input = input.clone();
    for (index, name) in tool.manifest.file_params.iter().enumerate() {
        let Some(value) = input.get(name).and_then(Value::as_str) else {
            continue;
        };
        if value.trim().is_empty() {
            continue;
        }
        let source = expand_path(value);
        let source = source
            .canonicalize()
            .with_context(|| format!("could not resolve {value}"))?;
        if !source.is_file() {
            bail!("{value} is not a file");
        }
        let file_name = source
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| {
                name.chars()
                    .map(|ch| {
                        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' || ch == '_' {
                            ch
                        } else {
                            '_'
                        }
                    })
                    .collect::<String>()
            })
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "file".to_string());
        let destination = PathBuf::from("/sandbox/incoming").join(format!("{index}-{file_name}"));
        add_bind(command_args, dirs, &source, &destination, false);
        input[name] = Value::String(destination.to_string_lossy().into_owned());
    }
    Ok(input)
}

async fn run_plan(plan: SandboxPlan) -> Result<ToolRun> {
    let mut command = Command::new(&plan.program);
    command
        .args(&plan.args)
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for (key, value) in &plan.env {
        command.env(key, value);
    }

    let mut child = command
        .spawn()
        .with_context(|| format!("could not start {}", Path::new(&plan.program).display()))?;

    if let Some(mut stdin) = child.stdin.take() {
        let payload = serde_json::to_vec(&plan.input).context("could not encode tool arguments")?;
        stdin
            .write_all(&payload)
            .await
            .context("could not send arguments to the tool")?;
        drop(stdin);
    }

    let stdout = child.stdout.take().context("tool has no stdout pipe")?;
    let stderr = child.stderr.take().context("tool has no stderr pipe")?;
    let max_output = plan.max_output_bytes;

    let out_future = async move {
        let mut reader = stdout.take((max_output + 1) as u64);
        let mut data = Vec::new();
        reader.read_to_end(&mut data).await.map(|_| data)
    };
    let err_future = async move {
        let mut reader = stderr.take(16 * 1024);
        let mut data = Vec::new();
        reader.read_to_end(&mut data).await.map(|_| data)
    };

    let outcome = tokio::time::timeout(plan.timeout, async {
        tokio::join!(out_future, err_future, child.wait())
    })
    .await;

    let (stdout, stderr, status) = match outcome {
        Ok((stdout, stderr, status)) => (stdout?, stderr?, status?),
        Err(_) => {
            let _ = child.start_kill();
            bail!("Tool timed out after {} seconds", plan.timeout.as_secs());
        }
    };

    let stdout_text = String::from_utf8_lossy(&stdout);
    let stderr_text = String::from_utf8_lossy(&stderr);
    let trimmed = stdout_text.trim();

    if trimmed.is_empty() {
        if !status.success() {
            let detail = stderr_text.trim();
            if detail.is_empty() {
                bail!("Tool exited with {status}");
            }
            bail!("Tool exited with {status}: {detail}");
        }
        return Ok(ToolRun {
            success: false,
            text: "Tool returned no output".to_string(),
        });
    }

    match serde_json::from_str::<Value>(trimmed) {
        Ok(Value::Object(map)) => {
            let success = map.get("success").and_then(Value::as_bool).unwrap_or(false);
            if success {
                let text = map
                    .get("result")
                    .and_then(Value::as_str)
                    .unwrap_or("(ok)")
                    .to_string();
                Ok(ToolRun {
                    success: true,
                    text: truncate_text(&text, max_output),
                })
            } else {
                let error = map
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                Ok(ToolRun {
                    success: false,
                    text: format!("Error: {error}"),
                })
            }
        }
        Ok(other) => Ok(ToolRun {
            success: true,
            text: truncate_text(&other.to_string(), max_output),
        }),
        Err(_) => Ok(ToolRun {
            success: true,
            text: truncate_text(trimmed, max_output),
        }),
    }
}

pub async fn execute(tool: &InstalledTool, args: &Value) -> Result<ToolRun> {
    let plan = build_sandbox_plan(tool, args)?;
    run_plan(plan).await
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut truncated: String = text.chars().take(max_chars).collect();
    truncated.push_str("…[truncated]");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> ToolManifest {
        ToolManifest {
            name: "demo_tool".to_string(),
            description: "A demo tool.".to_string(),
            version: 1,
            category: "core".to_string(),
            runtime: "python3".to_string(),
            requires: ToolRequires {
                commands: vec!["python3".to_string()],
                ..Default::default()
            },
            sandbox: ToolSandbox::default(),
            timeout_secs: 5,
            max_output_bytes: 4096,
            file_params: Vec::new(),
            dangerous_actions: Vec::new(),
            parameters: json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            }),
            default_permission: "ask".to_string(),
            default_cloud_permission: "deny".to_string(),
        }
    }

    fn tool(manifest: ToolManifest) -> InstalledTool {
        InstalledTool {
            manifest,
            dir: PathBuf::from("/tmp/does-not-matter"),
        }
    }

    #[test]
    fn rejects_path_traversal_ids() {
        assert!(is_valid_id("calculator"));
        assert!(is_valid_id("web_search"));
        assert!(!is_valid_id("../calculator"));
        assert!(!is_valid_id("Calculator"));
        assert!(!is_valid_id("a"));
        assert!(!is_valid_id("bad/id"));
        assert!(!is_valid_id("bad..id"));
    }

    #[test]
    fn validates_manifest_fields() {
        let mut manifest = manifest();
        assert!(validate_manifest("demo_tool", &manifest).is_ok());

        manifest.timeout_secs = 0;
        assert!(validate_manifest("demo_tool", &manifest).is_err());
        manifest.timeout_secs = 5;

        manifest.default_permission = "sometimes".to_string();
        assert!(validate_manifest("demo_tool", &manifest).is_err());
        manifest.default_permission = "ask".to_string();

        manifest.sandbox.read_paths = vec!["../../etc".to_string()];
        assert!(validate_manifest("demo_tool", &manifest).is_err());
    }

    #[test]
    fn permission_defaults_come_from_the_manifest() {
        let manifest = manifest();
        let settings = Settings::default();
        assert_eq!(
            permission_for(&settings, &manifest, false),
            ToolPermission::Ask
        );
        assert_eq!(
            permission_for(&settings, &manifest, true),
            ToolPermission::Deny
        );
    }

    #[test]
    fn sandbox_plan_requires_network_only_when_declared() {
        let mut manifest = manifest();
        let plan = build_sandbox_plan(&tool(manifest.clone()), &json!({})).unwrap();
        assert!(!plan.args.iter().any(|arg| arg == "--share-net"));

        manifest.sandbox.network = true;
        let plan = build_sandbox_plan(&tool(manifest), &json!({})).unwrap();
        assert!(plan.args.iter().any(|arg| arg == "--share-net"));
    }

    #[test]
    fn stages_declared_file_parameters() {
        let mut manifest = manifest();
        manifest.file_params = vec!["path".to_string()];
        manifest.parameters = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            }
        });
        let plan = build_sandbox_plan(
            &tool(manifest),
            &json!({"path": "/home/ltux/VSCodeProjects/Ollama-GUI-rs/Cargo.toml"}),
        )
        .unwrap();
        assert_eq!(
            plan.input.get("path").and_then(Value::as_str),
            Some("/sandbox/incoming/0-Cargo.toml")
        );
        assert!(plan.args.windows(3).any(
            |window| window[0] == "--ro-bind" && window[2] == "/sandbox/incoming/0-Cargo.toml"
        ));
    }

    #[test]
    #[ignore]
    fn executes_the_local_calculator_in_the_sandbox() {
        let dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Ollama-GUI-Tools/tools/calculator");
        let tool = load_installed(&dir).expect("load calculator");
        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let run = runtime
            .block_on(execute(&tool, &json!({"expression": "(2 + 3) * 4"})))
            .expect("execute calculator");
        assert!(run.success, "{}", run.text);
        assert!(run.text.contains("20"), "{}", run.text);
    }

    fn temp_tool(id: &str, script: &str, network: bool) -> (PathBuf, InstalledTool) {
        let root =
            std::env::temp_dir().join(format!("ollama-gui-sandbox-tests-{}", std::process::id()));
        let dir = root.join(id);
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&dir).expect("create temp tool");
        let manifest = json!({
            "name": id,
            "description": "Sandbox test tool",
            "version": 1,
            "runtime": "python3",
            "requires": {"commands": ["python3"]},
            "sandbox": {"network": network},
            "parameters": {"type": "object", "properties": {}},
            "default_permission": "allow",
            "default_cloud_permission": "allow"
        });
        std::fs::write(
            dir.join("tool.json"),
            serde_json::to_vec_pretty(&manifest).expect("manifest"),
        )
        .expect("write manifest");
        std::fs::write(dir.join("run"), script).expect("write run");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(dir.join("run"))
                .expect("metadata")
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(dir.join("run"), perms).expect("chmod");
        }
        let tool = load_installed(&dir).expect("load temp tool");
        (dir, tool)
    }

    #[test]
    #[ignore]
    fn sandbox_blocks_network_when_not_declared() {
        let script = r#"#!/usr/bin/env python3
import json
import urllib.request

try:
    urllib.request.urlopen("https://example.com", timeout=5)
    print(json.dumps({"success": True, "result": "reachable"}))
except Exception as exc:
    print(json.dumps({"success": False, "error": str(exc)}))
"#;
        let (dir, tool) = temp_tool("netblock", script, false);
        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let run = runtime
            .block_on(execute(&tool, &json!({})))
            .expect("execute");
        assert!(!run.success, "network should be blocked: {}", run.text);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    #[ignore]
    fn sandbox_allows_network_when_declared() {
        let script = r#"#!/usr/bin/env python3
import json
import urllib.request

urllib.request.urlopen("https://example.com", timeout=5)
print(json.dumps({"success": True, "result": "reachable"}))
"#;
        let (dir, tool) = temp_tool("netallow", script, true);
        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let run = runtime
            .block_on(execute(&tool, &json!({})))
            .expect("execute");
        assert!(run.success, "network should be available: {}", run.text);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    #[ignore]
    fn sandbox_hides_the_real_home_directory() {
        let home = crate::config::home_dir().join(".ssh");
        let script = r#"#!/usr/bin/env python3
import json
import os
import sys

params = json.load(sys.stdin)
print(json.dumps({
    "success": True,
    "result": "visible" if os.path.exists(params["path"]) else "hidden",
}))
"#;
        let (dir, tool) = temp_tool("homehide", script, false);
        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let run = runtime
            .block_on(execute(&tool, &json!({"path": home.to_string_lossy()})))
            .expect("execute");
        assert!(run.success, "{}", run.text);
        assert_eq!(run.text, "hidden");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn dangerous_actions_are_detected() {
        let mut manifest = manifest();
        manifest.dangerous_actions = vec!["session".to_string()];
        assert!(is_dangerous_action(
            &manifest,
            &json!({"action": "session"})
        ));
        assert!(!is_dangerous_action(
            &manifest,
            &json!({"action": "volume_up"})
        ));
    }
}
