use gtk::glib;
use gtk::prelude::*;
use libadwaita as adw;
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use sourceview5 as sourceview;
use sourceview5::prelude::*;

/// Render markdown into a read-only text view using native GTK text tags.
/// Fenced code blocks become embedded, syntax-highlighted source views.
pub fn render(view: &gtk::TextView, markdown: &str) {
    let buffer = view.buffer();
    buffer.set_text("");
    install_tags(&buffer);

    let parser = Parser::new_ext(
        markdown,
        Options::ENABLE_STRIKETHROUGH
            | Options::ENABLE_TABLES
            | Options::ENABLE_TASKLISTS
            | Options::ENABLE_GFM,
    );

    let mut active: Vec<&'static str> = Vec::new();
    let mut code: Option<(Option<String>, String)> = None;
    let mut lists: Vec<Option<u64>> = Vec::new();
    let mut item_depth = 0usize;

    for event in parser {
        match event {
            Event::Start(Tag::CodeBlock(kind)) => {
                let language = match kind {
                    CodeBlockKind::Fenced(info) => {
                        let first = info.split_whitespace().next().unwrap_or("");
                        (!first.is_empty()).then(|| first.to_string())
                    }
                    CodeBlockKind::Indented => None,
                };
                code = Some((language, String::new()));
            }
            Event::End(TagEnd::CodeBlock) => {
                if let Some((language, content)) = code.take() {
                    insert_code_block(view, &buffer, language, content);
                }
            }
            Event::Text(text) => {
                if let Some((_, content)) = code.as_mut() {
                    content.push_str(&text);
                } else {
                    insert(&buffer, &text, &active);
                }
            }
            Event::Code(text) => {
                insert(&buffer, &text, &["inline-code"]);
            }
            Event::SoftBreak => insert(&buffer, " ", &active),
            Event::HardBreak => insert(&buffer, "\n", &active),
            Event::Rule => insert(&buffer, "────────\n\n", &active),
            Event::Start(Tag::Paragraph) => {}
            Event::End(TagEnd::Paragraph) => {
                if item_depth > 0 {
                    insert(&buffer, "\n", &[]);
                } else {
                    insert(&buffer, "\n\n", &[]);
                }
            }
            Event::Start(Tag::Heading { level, .. }) => {
                active.push(match level {
                    HeadingLevel::H1 => "h1",
                    HeadingLevel::H2 => "h2",
                    HeadingLevel::H3 => "h3",
                    _ => "h4",
                });
            }
            Event::End(TagEnd::Heading(_)) => {
                active.pop();
                insert(&buffer, "\n\n", &[]);
            }
            Event::Start(Tag::Strong) => active.push("bold"),
            Event::End(TagEnd::Strong) => {
                active.pop();
            }
            Event::Start(Tag::Emphasis) => active.push("italic"),
            Event::End(TagEnd::Emphasis) => {
                active.pop();
            }
            Event::Start(Tag::Strikethrough) => active.push("strikethrough"),
            Event::End(TagEnd::Strikethrough) => {
                active.pop();
            }
            Event::Start(Tag::BlockQuote(_)) => {
                insert(&buffer, "│ ", &[]);
                active.push("quote");
            }
            Event::End(TagEnd::BlockQuote(_)) => {
                active.pop();
                insert(&buffer, "\n", &[]);
            }
            Event::Start(Tag::Link { .. }) => active.push("link"),
            Event::End(TagEnd::Link) => {
                active.pop();
            }
            Event::Start(Tag::Image { .. }) => insert(&buffer, "[image] ", &active),
            Event::End(TagEnd::Image) => {}
            Event::Start(Tag::List(start)) => lists.push(start),
            Event::End(TagEnd::List(_)) => {
                lists.pop();
                insert(&buffer, "\n", &[]);
            }
            Event::Start(Tag::Item) => {
                item_depth += 1;
                let indent = "  ".repeat(lists.len().saturating_sub(1));
                let marker = match lists.last_mut() {
                    Some(Some(number)) => {
                        let current = *number;
                        *number += 1;
                        format!("{current}. ")
                    }
                    _ => "• ".to_string(),
                };
                insert(&buffer, &format!("{indent}{marker}"), &active);
            }
            Event::End(TagEnd::Item) => {
                item_depth = item_depth.saturating_sub(1);
                insert(&buffer, "\n", &[]);
            }
            Event::TaskListMarker(checked) => {
                insert(&buffer, if checked { "☑ " } else { "☐ " }, &active);
            }
            Event::Start(Tag::Table(_)) | Event::Start(Tag::TableHead) => {}
            Event::End(TagEnd::TableHead) => insert(&buffer, "\n", &[]),
            Event::Start(Tag::TableRow) => {}
            Event::End(TagEnd::TableRow) => insert(&buffer, "\n", &[]),
            Event::Start(Tag::TableCell) => {}
            Event::End(TagEnd::TableCell) => insert(&buffer, " │ ", &[]),
            Event::End(TagEnd::Table) => insert(&buffer, "\n", &[]),
            Event::FootnoteReference(name) => {
                insert(&buffer, &format!("[^{name}]"), &active);
            }
            Event::Html(_) | Event::InlineHtml(_) => {}
            _ => {}
        }
    }

    trim_trailing_newlines(&buffer);
    view.queue_resize();
}

fn insert(buffer: &gtk::TextBuffer, text: &str, tags: &[&str]) {
    let mut end = buffer.end_iter();
    if tags.is_empty() {
        buffer.insert(&mut end, text);
    } else {
        buffer.insert_with_tags_by_name(&mut end, text, tags);
    }
}

fn trim_trailing_newlines(buffer: &gtk::TextBuffer) {
    let mut end = buffer.end_iter();
    while end.offset() > 0 {
        let mut start = end;
        start.backward_char();
        let ch = start.char();
        if ch == '\n' || ch == ' ' || ch == '\t' {
            buffer.delete(&mut start, &mut end);
            end = buffer.end_iter();
        } else {
            break;
        }
    }
}

fn install_tags(buffer: &gtk::TextBuffer) {
    let table = buffer.tag_table();

    let bold = ensure_tag(&table, "bold");
    bold.set_weight(700);

    let italic = ensure_tag(&table, "italic");
    italic.set_style(gtk::pango::Style::Italic);

    let strike = ensure_tag(&table, "strikethrough");
    strike.set_strikethrough(true);

    let inline = ensure_tag(&table, "inline-code");
    inline.set_family(Some("monospace"));

    let quote = ensure_tag(&table, "quote");
    quote.set_style(gtk::pango::Style::Italic);
    quote.set_left_margin(12);
    quote.set_foreground(Some("#7a7a7a"));

    let link = ensure_tag(&table, "link");
    link.set_underline(gtk::pango::Underline::Single);
    link.set_foreground(Some("#3584e4"));

    for (name, scale) in [("h1", 1.6), ("h2", 1.35), ("h3", 1.18), ("h4", 1.05)] {
        let tag = ensure_tag(&table, name);
        tag.set_scale(scale);
        tag.set_weight(700);
        tag.set_pixels_above_lines(6);
        tag.set_pixels_below_lines(2);
    }
}

fn ensure_tag(table: &gtk::TextTagTable, name: &str) -> gtk::TextTag {
    if let Some(tag) = table.lookup(name) {
        return tag;
    }
    let tag = gtk::TextTag::new(Some(name));
    table.add(&tag);
    tag
}

fn insert_code_block(
    view: &gtk::TextView,
    buffer: &gtk::TextBuffer,
    language: Option<String>,
    content: String,
) {
    let code = content.trim_end_matches('\n').to_string();

    let container = gtk::Box::new(gtk::Orientation::Vertical, 0);
    container.add_css_class("code-container");
    container.set_margin_top(6);
    container.set_margin_bottom(6);
    container.set_halign(gtk::Align::Fill);

    let header = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    header.set_margin_start(10);
    header.set_margin_end(6);
    header.set_margin_top(4);

    let language_label = gtk::Label::new(Some(language.as_deref().unwrap_or("code")));
    language_label.add_css_class("dim-label");
    language_label.set_xalign(0.0);
    language_label.set_hexpand(true);

    let copy_button = gtk::Button::from_icon_name("edit-copy-symbolic");
    copy_button.add_css_class("flat");
    copy_button.set_tooltip_text(Some("Copy code"));
    {
        let code = code.clone();
        copy_button.connect_clicked(move |button| {
            if let Some(display) = gtk::gdk::Display::default() {
                display.clipboard().set_text(&code);
            }
            button.set_icon_name("object-select-symbolic");
            let button = button.clone();
            glib::timeout_add_seconds_local_once(1, move || {
                button.set_icon_name("edit-copy-symbolic");
            });
        });
    }

    header.append(&language_label);
    header.append(&copy_button);
    container.append(&header);

    let source = sourceview::View::new();
    source.set_editable(false);
    source.set_cursor_visible(false);
    source.set_monospace(true);
    source.set_wrap_mode(gtk::WrapMode::WordChar);
    source.set_left_margin(10);
    source.set_right_margin(10);
    source.set_top_margin(4);
    source.set_bottom_margin(10);
    source.add_css_class("code-block");

    let source_buffer = sourceview::Buffer::new(None);

    if let Some(language) = &language {
        let manager = sourceview::LanguageManager::default();
        let lookup = match language.as_str() {
            "rs" => "rust",
            "py" => "python",
            "js" => "javascript",
            "ts" => "typescript",
            "sh" | "shell" => "bash",
            "yml" => "yaml",
            "md" => "markdown",
            other => other,
        };
        if let Some(language) = manager.language(lookup) {
            source_buffer.set_language(Some(&language));
        }
    }

    let scheme_manager = sourceview::StyleSchemeManager::default();
    let dark = adw::StyleManager::default().is_dark();
    let scheme = if dark {
        scheme_manager
            .scheme("Adwaita-dark")
            .or_else(|| scheme_manager.scheme("Adwaita"))
    } else {
        scheme_manager.scheme("Adwaita")
    };
    if let Some(scheme) = scheme {
        source_buffer.set_style_scheme(Some(&scheme));
    }

    source_buffer.set_text(&code);
    source.set_buffer(Some(&source_buffer));
    container.append(&source);

    let mut end = buffer.end_iter();
    let anchor = buffer.create_child_anchor(&mut end);
    view.add_child_at_anchor(&container, &anchor);
    insert(buffer, "\n\n", &[]);
}
