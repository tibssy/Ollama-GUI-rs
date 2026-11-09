# Ollama GUI

A native GTK4 + libadwaita desktop client for [Ollama](https://ollama.com),
written in Rust. Linux-first, Wayland-friendly, and built around the GNOME HIG.

## Status

The MVP is working:

- Streaming chat against a local or remote Ollama server
- Stop generation, per-reply token-per-second stats
- Per-message actions: copy, remove a turn from history, regenerate replies, and
  read assistant replies aloud (generated sentence by sentence and played in the
  background with rodio, so playback starts before the whole reply is synthesized)
- Local Paradee ONNX text-to-speech with a language setting, plus a preferred-TTS
  switch between Paradee and the selected model's built-in Ollama speech voices
- Local Whistle ONNX speech-to-text with Silero VAD, a microphone button,
  language selection, configurable start/end-silence timeouts, and optional
  automatic sending of the transcript
- Parakeet Redux ONNX speech-to-text as the preferred local provider: 25
  languages with automatic language detection
- Continuous live voice conversation with voice interruption (talk over the
  assistant) and automatic multi-turn sending/reading
- Context usage indicator (tokens used / window size, percentage) with a
  per-conversation percentage in the sidebar
- Collapsible sidebar with a header toggle, menu item, and Ctrl+B shortcut
- Optional background mode with a system tray icon, live-state indicator, and
  tray actions for window visibility, live conversation, and quit
- Automatic context compression: at 85% usage, older turns are summarized while
  the last five turns stay verbatim
- Optional thinking mode for reasoning models, with on/off or low/medium/high
  controls and automatic minimization in live voice conversations
- Conversations persisted locally in SQLite
- New / delete chat, automatic titles from the first message
- Model picker with search, per-conversation model selection (chat-capable
  models only; decision and embedding models are hidden)
- Capability badges for the selected model (completion, vision, tools, thinking,
  audio, speech)
- Settings for server URL, API key, system prompt, temperature, context size,
  thinking, theme, text-to-speech, auto-read, speech language, voice, dictation,
  and VAD timings
- Native markdown rendering: headings, bold/italic/strikethrough, lists, quotes,
  tables, links, inline code, and fenced code blocks with copy buttons
- Code blocks use GtkSourceView and syntax highlighting when language specs are installed
- Follows the system light/dark theme

### Roadmap

- Model manager: pull with progress, delete, show model details
- Vision models: image attachments
- Tool-call and `think` blocks
- Chat search (SQLite FTS5), rename, export/import
- Flatpak packaging

## Requirements

- Rust 1.85+ (edition 2021)
- GTK 4.16+ and libadwaita 1.6+ development files
- GtkSourceView 5
- `espeak-ng` (used by the local Paradee voice model)
- A microphone with ALSA or PipeWire for dictation
- A running Ollama instance (`ollama serve`), version 0.3 or newer

On Arch Linux:

```sh
sudo pacman -S --needed rust gtk4 libadwaita gtksourceview5 alsa-lib espeak-ng
```

## Build and run

```sh
cargo run
```

Release build:

```sh
cargo build --release
```

The first build downloads the ONNX Runtime library through the `ort` crate.
The Paradee voice model (about 8.5 MB) is downloaded on first use, or ahead of
time from Preferences → Speech.

## Install locally

```sh
install -Dm755 target/release/ollama-gui ~/.local/bin/ollama-gui
install -Dm644 data/io.github.ltux.OllamaGui.desktop ~/.local/share/applications/io.github.ltux.OllamaGui.desktop
install -Dm644 data/io.github.ltux.OllamaGui.metainfo.xml ~/.local/share/metainfo/io.github.ltux.OllamaGui.metainfo.xml
install -Dm644 data/icons/io.github.ltux.OllamaGui.svg ~/.local/share/icons/hicolor/scalable/apps/io.github.ltux.OllamaGui.svg
install -Dm644 data/icons/io.github.ltux.OllamaGui-live.svg ~/.local/share/icons/hicolor/scalable/apps/io.github.ltux.OllamaGui-live.svg
update-desktop-database ~/.local/share/applications 2>/dev/null || true
gtk-update-icon-cache -f -t ~/.local/share/icons/hicolor 2>/dev/null || true
```

## Configuration

Settings are stored at `$XDG_CONFIG_HOME/ollama-gui/settings.json`
(`~/.config/ollama-gui/settings.json` by default). Conversations live in
`$XDG_DATA_HOME/ollama-gui/chats.db` (`~/.local/share/ollama-gui/chats.db`).
The Paradee model is cached in `$XDG_DATA_HOME/ollama-gui/models/paradee/`.
The speech-to-text models are cached in
`$XDG_DATA_HOME/ollama-gui/models/whistle/` (about 160 MB on first use) and
`$XDG_DATA_HOME/ollama-gui/models/parakeet-redux/` (about 420 MB on first use).

The server URL defaults to `http://localhost:11434`. For a remote or Ollama
Cloud endpoint, set the URL and, if needed, an API key under Preferences.

## Speech

Preferences → Speech has the following choices:

- **Enable text-to-speech** — show the read-aloud buttons on assistant replies
- **Preferred TTS**
  - *Automatic* — Paradee for English, the selected Ollama model for other languages
  - *Paradee (local ONNX)*
  - *Ollama model (built-in)*
- **Language** — English (US/UK) for Paradee; the built-in model offers the
  languages its voices support (19 languages with `lfm3-audio`: Arabic, Chinese,
  Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Italian,
  Japanese, Korean, Polish, Portuguese, Spanish, Swedish, Turkish).
- **Voice** — the voice list follows the effective backend and language:
  Paradee exposes four voices (`af_heart`, `am_puck`, `bf_isabella`,
  `bm_fable`), while the built-in backend exposes a female and male voice for
  each language (e.g. `female-spanish` / `male-spanish`).
- **Read replies automatically** — speak each assistant reply as it streams in,
  starting with the first complete sentence; suppressed when text-to-speech is
  disabled.

Paradee 8M v1.2 is a small English-only model distilled from Kokoro-82M with
four voices. The int8 ONNX file is downloaded on first use (or from the
**Paradee model → Download** button) from
[huggingface.co/sahilmahendrakar/Paradee-8M-v1.2](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2)
and runs locally through ONNX Runtime. British voices use British eSpeak
phonemes. The model is Apache-2.0; see its model card for details.

Speech is synthesized one sentence at a time and queued for background playback,
so the first sentence starts before the rest of the reply is generated. Before
synthesis, markdown formatting, fenced code blocks, link targets, and
symbols/emoji are stripped so only speakable text is sent to the engine.

## Speech to Text

Preferences → Speech to Text enables the microphone button in the composer.

- **Preferred STT** — *Automatic* prefers Parakeet Redux when its model is
  downloaded, then Whistle when its model is downloaded and the language is
  supported, then the selected Ollama model when it supports audio input, and
  otherwise Whistle. Parakeet Redux, Whistle, and the built-in model can also be
  forced explicitly.
- **Language** — Parakeet Redux detects the language automatically. Whistle
  supports English, German, French, Spanish, Italian, Dutch and Polish.
  Built-in languages depend on the selected model (`lfm3-audio` supports 19
  languages). "Auto detect" is available.
- **Wait for speech** — if no voice is detected within this many seconds,
  listening stops and the microphone becomes clickable again.
- **End of speech silence** — seconds of silence after which the utterance is
  finalized.
- **Send automatically** — optionally send the transcript to the model as soon
  as speech ends. When disabled, the text stays in the composer.

Audio is captured at the device rate, mixed to mono, resampled to 16 kHz, and
processed in 512-sample frames by the bundled Silero VAD model. While speech is
active, partial transcripts update the input field every ~1.5 seconds; the final
transcript replaces the partial text when the end-of-speech silence is detected.

## Context compression

When a reply pushes context usage to 85% of the configured **Context size**,
the app summarizes older turns in the background and keeps the last five user
turns verbatim. The summary is stored with the conversation and reused
incrementally on later turns, so it is not regenerated on every request.
Original messages remain stored and visible in the chat.

An **Earlier messages summarized** row above the conversation opens the current
summary. If summarization fails, the next request falls back to the most recent
turns within the context budget.

## Sidebar

Use the sidebar button in the header, **Toggle Sidebar** in the main menu, or
Ctrl+B to hide or show the conversation list. The choice is remembered between
sessions. The chat column keeps its width when the sidebar is hidden, so
toggling the sidebar does not re-wrap long conversations.

## Background mode

Enable **Keep running in the background** under Preferences → Application to
keep the app alive after the window is closed. Closing the window hides it to
the system tray; use the tray icon or the **Show/Hide Window** tray action to
bring it back.

The tray menu can show or hide the window, start or stop live conversation
without opening the window, and quit the app. The tray icon changes when live
voice conversation is active.

The tray uses the freedesktop StatusNotifierItem specification, so a compatible
panel or host is required (KDE, GNOME's AppIndicator extension, Waybar, and
similar).

## Thinking

For models that advertise the `thinking` capability, a **Think** toggle appears
above the composer. When enabled, reasoning streams into the assistant bubble
behind a collapsible **Thinking** row with a spinner and chevron button. The row
can be expanded to read the reasoning and collapsed again with the same button.
The toggle choice is remembered between sessions.

Models with named reasoning levels show a level selector instead of an on/off
toggle. For example, gpt-oss exposes **Low**, **Medium**, and **High**; it cannot
fully disable reasoning, so the lowest level is used during live conversation.
Other thinking models use the normal on/off toggle, and live conversation turns
it off.

## Live voice conversation

A dedicated live-conversation button sits next to the dictation microphone in
the composer. It is available when both **Enable dictation** and
**Enable text-to-speech** are on.

- Starts a continuous STT session that keeps listening across turns.
- Each finished utterance is sent to the model automatically.
- Assistant replies are spoken automatically as they stream: each complete
  sentence is synthesized and queued while the model is still generating, so
  playback starts after the first sentence arrives.
- Assistant replies are spoken automatically, even when **Read replies
  automatically** is off.
- **Voice interruption:** speaking while the assistant is generating or talking
  stops the current reply/playback and starts the next turn. Interrupted text
  replies are kept in history but are not read aloud.
- The dictation microphone is disabled while live conversation is active.
- Stopping live conversation ends listening and stops any current speech; an
  in-flight text reply is allowed to finish.

> **Recommended for live mode:** use a headset. With speakers, enable echo
> cancellation on the microphone input first; otherwise the assistant's own
> voice can be picked up and trigger barge-in or repeated turns. A working
> EasyEffects input chain is:
>
> - **Echo Canceller** (WebRTC) enabled, with **Enforce high-pass** on
> - **Noise Suppression** at **Very high**
> - **High-Pass Filter** enabled on **Full band**

During live conversation, the speech-to-text VAD start timeout is ignored;
listening continues until the live button is turned off.

Parakeet Redux is an ONNX export of the 1.58-bit
[moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux)
model (based on NVIDIA's `parakeet-tdt-0.6b-v3`) from
[eschmidbauer/parakeet-redux-onnx](https://huggingface.co/eschmidbauer/parakeet-redux-onnx).
It handles 25 European languages and detects the language on its own, so its
language setting is always "Auto detect". The first use downloads about 420 MB
of model files (CC-BY-4.0). It runs on the CPU through ONNX Runtime with greedy
token-and-duration decoding and shares the same Silero VAD and microphone
pipeline as Whistle.

Whistle is a 16.9 MB Apache-2.0 model from Cactus Compute, using the ONNX export
from [mrfakename/whistle-ONNX](https://huggingface.co/mrfakename/whistle-ONNX)
(Apache-2.0). The first use downloads the encoder/decoder and pack (~160 MB).
The built-in provider uses Ollama's OpenAI-compatible
`/v1/audio/transcriptions` endpoint with the selected model. A microphone and
working ALSA/PipeWire input are required.

### Model footprint

| Component | On-disk size |
|---|---|
| Paradee int8 ONNX + config | 8.5 MB |
| Silero VAD (bundled) | 2.3 MB |
| Whistle ONNX export (encoder + decoder + pack) | ~156 MB |
| Parakeet Redux ONNX (preprocessor + encoder + decoder + vocab) | ~420 MB |
| Built-in TTS/STT | the selected Ollama model, e.g. 1.7 GB for `lfm3-audio` |

Models are loaded lazily. Repeated speech requests do not grow the process
without bound: ONNX Runtime arenas reach a high-water mark, and the Whistle
engram cache is capped at 16,384 rows (about 8 MB). Local engines are released
after 60 seconds without speech or dictation activity, and immediately when
text-to-speech or dictation is disabled; both paths trim the heap, and the next
use reloads the engine lazily. Measured warm RSS is about 230 MB for Paradee,
315 MB for Whistle, 580 MB for Parakeet Redux loaded (about 600 MB after a
transcription), or about 385 MB for Paradee and Whistle together. Unloading
Parakeet Redux and trimming the heap returns the process to about 50 MB.

## Architecture

```
src/
  main.rs            GtkApplication bootstrap, CSS, app actions
  config.rs          JSON settings in the XDG config dir
  api/mod.rs         thin Ollama client + streaming worker
  api/stream.rs      incremental NDJSON parser (unit tested)
  store/mod.rs       SQLite schema, migrations, conversation CRUD
  paradee.rs         Paradee download, eSpeak G2P conversion, ONNX inference
  whistle.rs         Whistle pack parser, mel frontend, engram cache,
                     ONNX encoder/decoder inference
  parakeet.rs        Parakeet Redux download, log-mel preprocessor, ONNX
                     encoder/decoder and greedy TDT decoding
  stt.rs             microphone capture, resampling, Silero VAD and the
                     streaming transcription worker
  tray.rs            StatusNotifierItem tray, icon state and menu actions
  ui/window.rs       split-view shell, chat controller, settings dialog,
                     sentence queuing and background speech playback
  ui/markdown.rs     pulldown-cmark -> native GTK tags + GtkSourceView blocks
resources/style.css  libadwaita-aware styling
data/                desktop entry, AppStream metadata, icon
```

Streaming runs on a background tokio runtime. Parsed events travel to the GTK
main loop through an `async_channel`; widgets are only touched from the main
thread. Stopping generation aborts the tokio task and keeps partial output.

## Tests

```sh
cargo test
```

The live end-to-end test needs a local Ollama daemon and at least one model:

```sh
cargo test -- --ignored --nocapture
```
