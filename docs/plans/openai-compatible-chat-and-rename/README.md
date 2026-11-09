# Feature Plan — OpenAI-Compatible Chat Providers + Local Embeddings + App Rename

Status: **plan only** — no implementation in this branch.
Date: 2026-11-09
Target branch: `feature/openai-compatible-chat-plan`
Branch contents: this README only. The root README and all source files are untouched.
Scope: provider matrix rows 1–2 only, plus the app/repository rename.

---

## 1. Goal

Let the app chat with either:

1. A native **Ollama** server (current behavior, unchanged), or
2. Any **OpenAI-compatible** chat provider — OpenRouter, Groq, LM Studio,
   llama.cpp server, vLLM, LocalAI, Jan, etc.

Long-term memory must keep using a **local Ollama embedding model only**. No
note text or retrieval query may be sent to a cloud embedding service.

In addition, rename the application and the GitHub repository so the product
name no longer implies it is Ollama-only, and migrate existing user data.

## 2. Non-goals

- Cloud embeddings of any kind.
- Native Anthropic, Gemini, or Azure OpenAI integrations.
- Per-provider UI or per-provider code paths.
- OpenAI-compatible TTS/STT as the chat provider. Built-in Ollama audio keeps
  using the local Ollama server; Parakeet, Whistle, and Piper are unaffected.
- Mapping thinking levels to `reasoning_effort` in the first pass. The Think
  control is hidden for OpenAI-compatible providers.
- Implementing any of this in the plan branch. The branch contains this
  document only.

## 3. Supported provider matrix

| Chat backend | Embeddings | Memory pool | Status |
|---|---|---|---|
| Local Ollama | Local Ollama | Local | Implement |
| Remote Ollama / Ollama Cloud | Local Ollama | Cloud | Implement |
| OpenAI-compatible (remote) | Local Ollama | Cloud | Implement |
| OpenAI-compatible (local server) | Local Ollama | Local | Implement |
| Any | Cloud embeddings | — | **Rejected** |

The memory pool follows the **chat** backend only. The embedding backend never
changes the pool.

## 4. Model discovery

### 4.1 Endpoints

| API style | Base URL the user enters | List request |
|---|---|---|
| Ollama | `http://localhost:11434` | `GET /api/tags` |
| OpenRouter | `https://openrouter.ai/api/v1` | `GET /v1/models` |
| Groq | `https://api.groq.com/openai/v1` | `GET /openai/v1/models` |
| OpenAI | `https://api.openai.com/v1` | `GET /v1/models` |
| LM Studio | `http://localhost:1234/v1` | `GET /v1/models` |
| llama.cpp | `http://localhost:8080/v1` | `GET /v1/models` |
| vLLM | `http://localhost:8000/v1` | `GET /v1/models` |

Rule: the stored base URL includes the version segment. The client appends only
`/models`, `/chat/completions`, etc. Do not silently insert `/v1`; instead show
a hint when the URL looks wrong (`404` → “Check that the URL ends with /v1”).

### 4.2 Authentication

- Send `Authorization: Bearer <api_key>` when a key is configured.
- OpenRouter’s model list is public, so listing works before a key is entered.
- Groq returns `401 {"error":{"message":"Invalid API Key",...}}` without a key.
- Local servers usually ignore the header.

### 4.3 Response shape

Standard OpenAI shape:

```json
{
  "object": "list",
  "data": [
    { "id": "llama-3.3-70b-versatile", "object": "model", "created": 1693721698, "owned_by": "Meta" }
  ]
}
```

Provider extras to consume when present:

- OpenRouter: `name`, `context_length`, `architecture.input_modalities`,
  `architecture.output_modalities`, `pricing`, `supported_parameters`,
  `:free` model IDs.
- Groq: `active`, `context_window`, `owned_by`.
- Local servers: often only `id`.

### 4.4 Internal model type

Extend `ModelSummary` in `src/api/mod.rs`:

```rust
pub struct ModelSummary {
    pub name: String,
    pub size: u64,
    pub details: ModelDetails,
    pub capabilities: Vec<String>,
    pub remote_host: Option<String>,      // Ollama Cloud only
    pub context_length: Option<u32>,      // OpenRouter context_length / Groq context_window
    pub owned_by: Option<String>,         // display only
}
```

Do **not** reuse `remote_host` for OpenAI providers. Pool detection already
treats a non-loopback chat base URL as remote.

### 4.5 Filtering non-chat models

OpenRouter:

- Keep models whose `architecture.input_modalities` contains `text` **and**
  `architecture.output_modalities` contains `text`.
- This excludes image generation and embedding models cleanly.

Groq / generic providers:

- Exclude IDs containing `whisper`, `tts`, `speech`, `embed`, `rerank`,
  `moderation`, `guard`, `dall-e`.
- Exclude `active == false` when the field exists.
- Keep unknown IDs rather than over-filtering.

Infer `capabilities` for OpenAI models as `["completion"]` when they pass the
chat filter. Do not claim `vision`, `tools`, or `thinking` unless the provider
explicitly reports them.

### 4.6 Model picker UX

- A plain dropdown is unusable for OpenRouter (currently 464 models).
- The app targets GTK 4.24, so use `DropDown::set_enable_search(true)` and a
  search placeholder (“Search models…”).
- Label entries with the model ID and, when known, context size:
  `anthropic/claude-… · 200k`.
- Mark `:free` entries, but do not sort or default to them.
- Show an empty state that distinguishes “no models” from “connection failed”.

### 4.7 Refresh behavior

- Fetch on startup, on settings save, and on the manual refresh button.
- Do not poll.
- Keep the previous list visible while a refresh is in flight.
- Cache nothing on disk; the provider is the source of truth.

### 4.8 Error handling

- Parse `{"error":{"message":…,"type":…,"code":…}}` from non-2xx responses.
- 401 → “Invalid API key for <host>”.
- 404 → “Check that the base URL ends with /v1”.
- 429 → “Rate limited; try again later”.
- Network failure → “Could not reach <host>”.
- Never print the API key.

## 5. Chat transport

### 5.1 Request mapping

| Field | Ollama | OpenAI-compatible |
|---|---|---|
| Endpoint | `POST /api/chat` | `POST /chat/completions` |
| Messages | `messages[]` | `messages[]` |
| Stream | `stream: true` | `stream: true` |
| Temperature | `options.temperature` | `temperature` |
| Output limit | `options.num_predict` | `max_tokens` / `max_completion_tokens` |
| Context size | `options.num_ctx` | not sent |
| Keep alive | `keep_alive` | not sent |
| Thinking | `think` | not sent in v1 |
| Usage | `eval_count`, `eval_duration` | `usage` in final chunk (optional) |

Add an `ApiStyle` enum (`Ollama`, `OpenAi`) and dispatch inside the client.
Keep the public `ChatRequest`/`ChatMessage` types provider-neutral; build the
wire body per style.

### 5.2 Streaming parser

OpenAI-compatible providers stream SSE, not NDJSON:

```
data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world"}}]}

data: [DONE]
```

- Reuse the existing line buffer, but add a small SSE decoder that strips
  `data:` prefixes, ignores comments/`event:` lines, and stops at `[DONE]`.
- Read `choices[0].delta.content`.
- Optionally read `choices[0].delta.reasoning` / `reasoning_content` into
  `StreamEvent::Thinking`; if the provider does not send it, nothing changes.
- Read final `usage` when present. Token/sec stats may be unavailable; show
  “—” instead of inventing numbers.
- Handle a stream that ends without `[DONE]` by emitting `Done` with whatever
  stats were collected.

### 5.3 Stop generation

The existing abort handle works unchanged because the HTTP response stream is
aborted. Verify the OpenAI path sends no trailing request after abort.

## 6. Settings

Add to `Settings` in `src/config.rs`, all with serde defaults so old files load:

```rust
/// "ollama" or "openai"
pub api_style: String,                 // default "ollama"
/// Local Ollama used for embeddings and built-in audio
pub embedding_base_url: String,        // default "http://localhost:11434"
/// Optional key for the local Ollama server
pub embedding_api_key: String,         // default ""
```

`memory_embedding_model` stays as-is.

### 6.1 Preferences UI

Connection group:

- API style: ComboRow — “Ollama”, “OpenAI-compatible”.
- Server URL: placeholder changes by style:
  - Ollama: `http://localhost:11434`
  - OpenAI: `https://openrouter.ai/api/v1`
- API key (optional): unchanged.
- Inline hint when the style is OpenAI: “Enter the full base URL, including /v1.”

Local services group:

- Local Ollama URL (embeddings + built-in audio), default
  `http://localhost:11434`.
- Optional local API key.
- Embedding model ComboRow, populated from the local Ollama model list.
- Note: “Memory notes and queries are embedded on this server. Keep it local.”

Generation group:

- In OpenAI mode, disable or relabel Context size: “Context size (Ollama only)”.
- Keep Temperature.
- Compression uses the provider’s `context_length` when known.

### 6.2 Validation

- Parse both URLs with `reqwest::Url`.
- If `embedding_base_url` is a public host, show a warning and disable memory
  until it is changed. Treat loopback, RFC1918, and link-local addresses as
  local.
- If the local Ollama server is unreachable, chat still works; memory is paused
  with a toast, not a hard failure.

## 7. Client architecture

### 7.1 Two clients

- `state.client` — chat, chat model list, `/api/show`, compression, memory
  extraction.
- `state.embed_client` — local Ollama only: `/api/embed`, plus built-in
  `/v1/audio/speech` and `/v1/audio/transcriptions` so voice keeps working when
  chat is remote.

Both are rebuilt when settings are saved.

### 7.2 Internal naming

`OllamaClient` is no longer accurate once it can speak OpenAI. Rename it to
`ApiClient` and keep the type private to `src/api/mod.rs` where possible. This
is a mechanical rename; do it in the same commit as the provider abstraction so
the diff stays coherent.

### 7.3 Dispatch points

- `list_models()` → `/api/tags` or `/models`, normalized to `ModelSummary`.
- `chat()` / `spawn_chat_stream()` → NDJSON or SSE decoder.
- `model_thinking()` → Ollama only; return `None` for OpenAI.
- `embed()` → only on the local client.
- `speech()` / `transcribe()` → only on the local client.

### 7.4 Memory job

`run_memory_job` currently takes one client. Change it to take:

```rust
run_memory_job(chat_client, embed_client, job)
```

- Extraction prompt and `chat_once` use `chat_client`.
- `embed()` uses `embed_client`.
- The resulting note is stored in the pool derived from the chat model.
- If `embed_client` fails, advance the checkpoint as today and show a warning.

### 7.5 Embedding model discovery

- `embedding_model_ready()` and the Embedding model ComboRow must read from
  `embed_client.list_models()`, not the chat model list.
- Store the local list in `state.embedding_models` and refresh it separately.
- If the local server is down, show the existing “install an embedding model”
  guidance but distinguish “server unreachable” from “model missing”.

## 8. Memory and privacy behavior

- Pool decision: chat base URL, chat model suffix, and Ollama `remote_host`
  only. Never the embedding URL.
- Local embeddings are mandatory. Reject public embedding hosts.
- Existing store keying (`pool` + `embedding_model`) is sufficient; no schema
  change.
- `search_entries` already ignores vectors with a different length, so changing
  the local embedding model cannot mix vector spaces.
- Cloud chat + local embeddings: the conversation already goes to the chat
  provider; note text and retrieval queries stay on the local machine.
- Local chat + local embeddings: nothing leaves the machine.
- Document the privacy matrix in the README.

## 9. Thinking and capabilities

- Ollama: keep `/api/show` behavior, boolean and level-only models.
- OpenAI-compatible: `model_thinking()` returns `None`; hide the Think toggle
  and level selector; never send `think`.
- Do not show Ollama-specific capability chips for OpenAI models.
- Later extension: map to `reasoning_effort` (Groq) or `reasoning.effort`
  (OpenRouter), behind a provider-quirk layer. Out of scope for v1.

## 10. Compression and context

- `num_ctx` remains the Ollama context setting.
- For OpenAI models with `context_length`, use that value as the effective
  context for compression and memory budget calculations.
- If the provider reports no context length, fall back to the `num_ctx` setting.
- Update the “Estimated from the last Ollama reply” tooltip to be
  provider-neutral.

## 11. Audio

- Parakeet, Whistle, Piper, and the local audio engines are unchanged.
- Built-in Ollama TTS/STT use `embed_client` (the local Ollama server), so they
  still work when chat is Groq/OpenRouter.
- If the local server has no audio model, show the existing capability-based
  error; do not fall back to the remote chat provider.

## 12. App and repository rename

### 12.1 Is it possible?

Yes. GitHub repository renames keep redirects for the old URL, and a local
`git remote set-url` updates the clone. The application rename is mechanical
but touches identifiers that must stay consistent:

- GTK application ID / D-Bus name
- desktop file basename and `Icon=`
- AppStream metainfo `id` and `launchable`
- icon filenames
- tray StatusNotifier ID and icon names
- Cargo package and binary name
- XDG config and data directories
- README install commands and paths

The one real risk is user data: changing `config_dir()`/`data_dir()` hides
existing settings, conversations, memories, and downloaded voice models unless
a migration runs.

### 12.2 Naming constraints

Pick one name and derive all forms:

| Form | Example | Notes |
|---|---|---|
| Display name | `Hearth` | Window title, About, tray |
| Crate/binary | `hearth` | kebab-case, no spaces |
| App ID | `io.github.tibssy.Hearth` | reverse-DNS, PascalCase |
| Desktop file | `io.github.tibssy.Hearth.desktop` | must match app ID |
| Metainfo | `io.github.tibssy.Hearth.metainfo.xml` | `<id>` matches |
| Icons | `io.github.tibssy.Hearth.svg`, `…-live.svg` | basename matches |
| Repo | `Hearth` or `Hearth-rs` | GitHub |
| Config dir | `~/.config/hearth/` | kebab-case |
| Data dir | `~/.local/share/hearth/` | kebab-case |

Notes:

- The current app ID uses `ltux`, but the GitHub owner is `tibssy`. Fix the
  namespace during the rename.
- Do not put “Ollama” in the product name. Descriptive references such as
  “works with Ollama” in the README are fine, but the name should not imply
  affiliation or Ollama-only scope.
- Check trademark, domain, and GitHub availability before committing to a name.
- Candidate directions (not decided): Hearth, Lantern, Locus, Cove, Myna.
- The repository name may differ from the app name; `-rs` is optional.

### 12.3 Touchpoint inventory

| Area | File / location | Change |
|---|---|---|
| Crate + binary | `Cargo.toml` `[package] name` | new kebab-case name |
| Lockfile | `Cargo.lock` | regenerated by build |
| App ID | `src/main.rs` `.application_id(...)` | new reverse-DNS ID |
| Window title | `src/main.rs` error window, `src/ui/window.rs` `.title(...)` | display name |
| About dialog | `src/ui/window.rs` `show_about()` | name, icon, comments |
| Menu | `src/ui/window.rs` “About …” | display name |
| Status / empty state | `src/ui/window.rs` tooltips and empty-state copy | provider-neutral, new name |
| Tray | `src/tray.rs` `id`, `title`, `icon_name`, tooltip, menu | new ID + names |
| XDG paths | `src/config.rs` `config_dir()`, `data_dir()` | new kebab-case dirs |
| Desktop entry | `data/*.desktop` | filename, `Name`, `Exec`, `Icon` |
| AppStream | `data/*.metainfo.xml` | filename, `id`, `name`, `launchable` |
| Icons | `data/icons/*.svg` | filenames |
| README | title, install commands, paths, clone URL | new names |
| Tests | `src/api/mod.rs` test strings “Hello from Ollama GUI.” | new name |
| Git remote | `origin` | new repo URL |
| GitHub | repo name, description, topics | new name |
| User install | `~/.local/bin`, applications, metainfo, icons | install new, remove old |
| Compositor | user niri/sway window rules keyed to app ID | release note |

### 12.4 Data migration

Run a one-time, best-effort migration before `Settings::load()`:

1. If the new config dir is missing and the old `~/.config/ollama-gui/` exists,
   move it to the new location. Preserve `0600` on `settings.json`.
2. If the new data dir is missing and `~/.local/share/ollama-gui/` exists,
   move it. A directory rename on the same filesystem is cheap even for the
   ~580 MB voice-model caches.
3. If both old and new exist, do not merge or overwrite. Log a warning and use
   the new directory.
4. If the move fails, leave the old directory untouched and continue with a
   fresh new directory. Never delete user data.
5. Keep `chats.db` as the database filename; only its parent changes.
6. Make the migration idempotent and covered by a unit test using temp dirs.

The migration must be an explicit function called from `main()` (for example
`config::migrate_legacy_paths()`), not a side effect inside `config_dir()`.

### 12.5 Install / uninstall

- Install the new binary, desktop file, metainfo, and both icons.
- Remove the old desktop file, metainfo, and icons so the launcher does not
  show two entries.
- Refresh caches:
  `update-desktop-database ~/.local/share/applications` and
  `gtk-update-icon-cache ~/.local/share/icons/hicolor` when available.
- Document the old binary path in the release notes; do not delete it
  automatically.

### 12.6 Repository rename steps

1. Choose the final name and confirm availability.
2. Rename on GitHub (`gh repo rename <new-name>` or Settings → Rename).
3. Update `origin`:
   `git remote set-url origin https://github.com/tibssy/<new-name>.git`.
4. Verify `git fetch` and `git push` still work; old URLs redirect.
5. Update README clone/install URLs and any issue links.
6. Update the local folder name only if desired; it is not required.
7. Rebuild, install, and smoke-test the renamed app and tray.

### 12.7 Rollback

- Keep the old config/data directories until the renamed app has been verified.
- If migration fails, restore by renaming the directories back.
- The old binary remains at its previous path until the user removes it.

## 13. Order of work

Recommended sequence:

1. **Decide the name** (blocks the rename phase only).
2. **Provider abstraction** — `ApiStyle`, request mapping, model discovery,
   SSE parser. Keep the Ollama path byte-for-byte equivalent.
3. **Local embedding split** — second client, settings, model discovery,
   memory job routing, privacy validation.
4. **UI and privacy polish** — searchable model picker, context length,
   remote-host warning, provider-neutral copy.
5. **Rename app + repo + migration** — last, so the new name reflects the
   broader provider support and the provider diff is already stable.
6. **Docs, tests, release.**

Rationale for renaming last: the provider work touches most of the same files
and already rewrites Ollama-specific UI copy. Renaming first would force a
second pass over the same lines and create avoidable conflicts.

## 14. Testing

### Unit tests

- OpenAI model-list parsing with fixtures for OpenRouter and Groq shapes.
- Chat filtering: OpenRouter modality, Groq audio/guard exclusion.
- SSE decoder: content deltas, reasoning deltas, `[DONE]`, missing `[DONE]`,
  split chunks, comments.
- Error-body parsing for 401/404/429.
- Base URL joining and the `/v1` hint.
- Private vs public host detection for the embedding URL.
- Settings defaults and legacy migration (temp dirs).
- Existing Ollama tests must remain green.

### Integration / manual

- Ollama-only regression: chat, thinking, memory, compression, TTS/STT.
- Local OpenAI-compatible server (LM Studio or llama.cpp) if available:
  model list, streaming chat, stop.
- OpenRouter with a key: model search across 464 entries, free-model labels,
  streaming.
- Groq with a key: model list filtering, streaming, 401 message.
- Memory: local chat → local pool; remote chat → cloud pool; vectors computed
  locally; public embedding URL rejected.
- Rename: fresh install, migration from old dirs, tray icon, desktop launch,
  About dialog, single instance.
- `cargo fmt --check`, `cargo build`, `cargo test`,
  `cargo clippy --all-targets`.

## 15. Branch and commit plan

The plan branch contains **only** this document, at
`docs/plans/openai-compatible-chat-and-rename/README.md`.

Branch: `feature/openai-compatible-chat-plan`.

Suggested commit:

```
docs: add OpenAI-compatible provider and rename plan
```

No source, config, asset, or README changes on this branch. Implementation
branches can be cut from `main` later:

- `feat/api-provider-abstraction`
- `feat/local-embeddings`
- `feat/model-picker-and-privacy`
- `chore/rename-app-and-repo`

## 16. Acceptance criteria

- [ ] Plan document is committed alone on a feature branch.
- [ ] `main` is unchanged.
- [ ] A native Ollama server behaves exactly as before.
- [ ] OpenRouter and Groq model lists load and are searchable.
- [ ] Streaming chat works with both API styles and can be stopped.
- [ ] Think controls are hidden for OpenAI-compatible providers.
- [ ] Embeddings always run on a local Ollama server.
- [ ] Public embedding URLs are rejected or disable memory with a warning.
- [ ] Memory pools follow the chat backend, not the embedding backend.
- [ ] Built-in Ollama audio keeps working with a remote chat provider.
- [ ] Rename migration preserves settings, conversations, memories, and voice
      model caches.
- [ ] Old desktop entry and icons are removed on install.
- [ ] README documents provider setup, the privacy matrix, and the new name.

## 17. Open questions

1. **New app name:** _(undecided — leave blank until a name is chosen)_
2. Should the repository keep a different name from the app (for example
   `<Name>-rs`)?
3. Should LAN Ollama servers count as “local” for embeddings? Recommendation:
   yes — loopback, RFC1918, and link-local.
4. Should OpenRouter `:free` models be visually highlighted? Recommendation:
   label them, do not sort or default to them.
5. Store the API key in the system keyring now, or keep the current `0600`
   `settings.json`? Recommendation: keyring as a follow-up.
6. Version bump: `0.2.0` for multi-provider, `1.0.0` for the rename, or a
   single release for both? Recommendation: one `0.2.0` release after both.
