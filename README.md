# Ollama GUI

A native GTK4 + libadwaita desktop client for [Ollama](https://ollama.com),
written in Rust. Linux-first, Wayland-friendly, and built around the GNOME HIG.

## Status

The MVP is working:

- Streaming chat against a local or remote Ollama server
- Stop generation, per-reply token-per-second stats
- Context usage indicator (tokens used / window size, percentage) with a
  per-conversation percentage in the sidebar
- Conversations persisted locally in SQLite
- New / delete chat, automatic titles from the first message
- Model picker with search, per-conversation model selection (chat-capable
  models only; decision and embedding models are hidden)
- Settings for server URL, API key, system prompt, temperature, context size, theme
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
- A running Ollama instance (`ollama serve`), version 0.3 or newer

On Arch Linux:

```sh
sudo pacman -S --needed rust gtk4 libadwaita gtksourceview5
```

## Build and run

```sh
cargo run
```

Release build:

```sh
cargo build --release
```

## Install locally

```sh
install -Dm755 target/release/ollama-gui ~/.local/bin/ollama-gui
install -Dm644 data/io.github.ltux.OllamaGui.desktop ~/.local/share/applications/io.github.ltux.OllamaGui.desktop
install -Dm644 data/io.github.ltux.OllamaGui.metainfo.xml ~/.local/share/metainfo/io.github.ltux.OllamaGui.metainfo.xml
install -Dm644 data/icons/io.github.ltux.OllamaGui.svg ~/.local/share/icons/hicolor/scalable/apps/io.github.ltux.OllamaGui.svg
update-desktop-database ~/.local/share/applications 2>/dev/null || true
```

## Configuration

Settings are stored at `$XDG_CONFIG_HOME/ollama-gui/settings.json`
(`~/.config/ollama-gui/settings.json` by default). Conversations live in
`$XDG_DATA_HOME/ollama-gui/chats.db` (`~/.local/share/ollama-gui/chats.db`).

The server URL defaults to `http://localhost:11434`. For a remote or Ollama
Cloud endpoint, set the URL and, if needed, an API key under Preferences.

## Architecture

```
src/
  main.rs            GtkApplication bootstrap, CSS, app actions
  config.rs          JSON settings in the XDG config dir
  api/mod.rs         thin Ollama client + streaming worker
  api/stream.rs      incremental NDJSON parser (unit tested)
  store/mod.rs       SQLite schema, migrations, conversation CRUD
  ui/window.rs       split-view shell, chat controller, settings dialog
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
