use std::cell::{Cell, RefCell};
use std::rc::Rc;

use gtk::prelude::*;
use gtk::{gdk, gio, glib};
use libadwaita as adw;
use libadwaita::prelude::*;

use crate::api::{
    self, ChatMessage, ChatOptions, ChatRequest, ModelSummary, OllamaClient, StreamEvent,
    StreamStats,
};
use crate::config::Settings;
use crate::store::Store;

pub struct AppState {
    pub settings: RefCell<Settings>,
    pub store: RefCell<Store>,
    pub runtime: tokio::runtime::Runtime,
    pub client: RefCell<OllamaClient>,
    pub models: RefCell<Vec<ModelSummary>>,
}

impl AppState {
    pub fn new() -> anyhow::Result<Self> {
        let settings = Settings::load();
        let client = OllamaClient::new(&settings.base_url, &settings.api_key);
        Ok(Self {
            settings: RefCell::new(settings),
            store: RefCell::new(Store::open()?),
            runtime: tokio::runtime::Runtime::new()?,
            client: RefCell::new(client),
            models: RefCell::new(Vec::new()),
        })
    }

    fn rebuild_client(&self) {
        let settings = self.settings.borrow();
        *self.client.borrow_mut() = OllamaClient::new(&settings.base_url, &settings.api_key);
    }
}

struct ActiveStream {
    abort: tokio::task::AbortHandle,
    row: gtk::Box,
    bubble: gtk::Box,
    view: gtk::TextView,
    content: Rc<RefCell<String>>,
    conv_id: i64,
    stats: Rc<RefCell<Option<StreamStats>>>,
}

struct ChatWindow {
    window: adw::ApplicationWindow,
    state: Rc<AppState>,

    chat_list: gtk::ListBox,
    new_chat_button: gtk::Button,
    refresh_button: gtk::Button,
    model_dropdown: gtk::DropDown,
    status_label: gtk::Label,

    title_widget: adw::WindowTitle,
    main_stack: gtk::Stack,
    messages_box: gtk::Box,
    messages_scroll: gtk::ScrolledWindow,
    input: gtk::TextView,
    send_button: gtk::Button,
    stop_button: gtk::Button,
    context_label: gtk::Label,
    context_bar: gtk::LevelBar,
    toast_overlay: adw::ToastOverlay,
    menu_button: gtk::MenuButton,

    current_id: RefCell<Option<i64>>,
    active: RefCell<Option<ActiveStream>>,
    suppress_model_change: Cell<bool>,
}

pub fn build(app: &adw::Application) -> anyhow::Result<adw::ApplicationWindow> {
    let state = Rc::new(AppState::new()?);
    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("Ollama GUI")
        .default_width(1080)
        .default_height(760)
        .build();

    let chat = ChatWindow::new(&window, state);
    chat.wire();
    chat.start();
    Ok(window)
}

impl ChatWindow {
    fn new(window: &adw::ApplicationWindow, state: Rc<AppState>) -> Rc<Self> {
        // ----- Sidebar -----------------------------------------------------
        let chat_list = gtk::ListBox::new();
        chat_list.add_css_class("navigation-sidebar");
        chat_list.set_selection_mode(gtk::SelectionMode::Single);
        chat_list.set_valign(gtk::Align::Start);

        let sidebar_scroll = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vexpand(true)
            .child(&chat_list)
            .build();

        let model_dropdown = gtk::DropDown::from_strings(&[]);
        model_dropdown.set_hexpand(true);
        model_dropdown.set_enable_search(true);
        model_dropdown.set_tooltip_text(Some("Model"));

        let status_label = gtk::Label::new(Some("Connecting…"));
        status_label.add_css_class("dim-label");
        status_label.set_xalign(0.0);
        status_label.set_wrap(true);
        status_label.set_tooltip_text(Some("Ollama connection status"));

        let footer = gtk::Box::new(gtk::Orientation::Vertical, 8);
        footer.set_margin_top(8);
        footer.set_margin_bottom(12);
        footer.set_margin_start(12);
        footer.set_margin_end(12);
        footer.append(&status_label);
        footer.append(&model_dropdown);

        let sidebar_content = gtk::Box::new(gtk::Orientation::Vertical, 0);
        sidebar_content.append(&sidebar_scroll);
        sidebar_content.append(&footer);

        let sidebar_header = adw::HeaderBar::new();
        let sidebar_title = gtk::Label::new(Some("Chats"));
        sidebar_title.add_css_class("title");
        sidebar_header.set_title_widget(Some(&sidebar_title));

        let new_chat_button = gtk::Button::from_icon_name("list-add-symbolic");
        new_chat_button.set_tooltip_text(Some("New chat"));
        new_chat_button.add_css_class("flat");

        let refresh_button = gtk::Button::from_icon_name("view-refresh-symbolic");
        refresh_button.set_tooltip_text(Some("Refresh models"));
        refresh_button.add_css_class("flat");

        let sidebar_actions = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        sidebar_actions.append(&refresh_button);
        sidebar_actions.append(&new_chat_button);
        sidebar_header.pack_end(&sidebar_actions);

        let sidebar_toolbar = adw::ToolbarView::new();
        sidebar_toolbar.add_top_bar(&sidebar_header);
        sidebar_toolbar.set_content(Some(&sidebar_content));

        // ----- Message area ------------------------------------------------
        let messages_box = gtk::Box::new(gtk::Orientation::Vertical, 12);
        messages_box.set_margin_top(18);
        messages_box.set_margin_bottom(18);
        messages_box.set_margin_start(18);
        messages_box.set_margin_end(18);

        let clamp = adw::Clamp::builder()
            .maximum_size(900)
            .tightening_threshold(600)
            .child(&messages_box)
            .build();

        let messages_scroll = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vexpand(true)
            .hexpand(true)
            .child(&clamp)
            .build();

        let empty_page = adw::StatusPage::builder()
            .title("Start a conversation")
            .description("Pick a model and say hello to your local Ollama.")
            .build();

        let main_stack = gtk::Stack::new();
        main_stack.set_transition_type(gtk::StackTransitionType::Crossfade);
        main_stack.add_named(&empty_page, Some("empty"));
        main_stack.add_named(&messages_scroll, Some("chat"));
        main_stack.set_visible_child_name("empty");
        main_stack.set_vexpand(true);

        // ----- Composer ----------------------------------------------------
        let input = gtk::TextView::new();
        input.set_wrap_mode(gtk::WrapMode::WordChar);
        input.set_accepts_tab(false);
        input.set_height_request(70);
        input.set_top_margin(4);
        input.set_bottom_margin(4);
        input.set_left_margin(6);
        input.set_right_margin(6);
        input.set_tooltip_text(Some("Enter to send, Shift+Enter for a new line"));

        let send_button = gtk::Button::from_icon_name("go-up-symbolic");
        send_button.add_css_class("suggested-action");
        send_button.add_css_class("circular");
        send_button.set_tooltip_text(Some("Send (Enter)"));

        let stop_button = gtk::Button::from_icon_name("media-playback-stop-symbolic");
        stop_button.add_css_class("destructive-action");
        stop_button.add_css_class("circular");
        stop_button.set_tooltip_text(Some("Stop generating"));
        stop_button.set_visible(false);

        let composer_buttons = gtk::Box::new(gtk::Orientation::Vertical, 6);
        composer_buttons.set_valign(gtk::Align::End);
        composer_buttons.append(&stop_button);
        composer_buttons.append(&send_button);

        let composer = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        composer.add_css_class("composer");
        composer.set_margin_top(6);
        composer.set_margin_bottom(12);
        composer.set_margin_start(12);
        composer.set_margin_end(12);
        input.set_hexpand(true);
        composer.append(&input);
        composer.append(&composer_buttons);

        let content_box = gtk::Box::new(gtk::Orientation::Vertical, 0);

        let context_label = gtk::Label::new(Some("Context: 0 / 8,192 tokens · 0%"));
        context_label.add_css_class("dim-label");
        context_label.set_xalign(0.0);
        context_label.set_hexpand(true);
        context_label.set_tooltip_text(Some(
            "Estimated from the last Ollama reply: prompt tokens + generated tokens",
        ));

        let context_bar = gtk::LevelBar::for_interval(0.0, 1.0);
        context_bar.set_value(0.0);
        context_bar.set_size_request(140, -1);
        context_bar.set_valign(gtk::Align::Center);
        context_bar.add_css_class("context-level");

        let context_row = gtk::Box::new(gtk::Orientation::Horizontal, 10);
        context_row.set_margin_start(18);
        context_row.set_margin_end(18);
        context_row.set_margin_top(4);
        context_row.append(&context_label);
        context_row.append(&context_bar);

        content_box.append(&main_stack);
        content_box.append(&context_row);
        content_box.append(&composer);

        let title_widget = adw::WindowTitle::new("New Chat", "No model selected");
        let header = adw::HeaderBar::new();
        header.set_title_widget(Some(&title_widget));

        let menu_button = gtk::MenuButton::new();
        menu_button.set_icon_name("open-menu-symbolic");
        menu_button.set_tooltip_text(Some("Main menu"));
        header.pack_end(&menu_button);

        let content_toolbar = adw::ToolbarView::new();
        content_toolbar.add_top_bar(&header);
        content_toolbar.set_content(Some(&content_box));

        let sidebar_page = adw::NavigationPage::builder()
            .title("Chats")
            .child(&sidebar_toolbar)
            .build();
        let content_page = adw::NavigationPage::builder()
            .title("Chat")
            .child(&content_toolbar)
            .build();

        let split = adw::NavigationSplitView::new();
        split.set_sidebar(Some(&sidebar_page));
        split.set_content(Some(&content_page));
        split.set_min_sidebar_width(240.0);
        split.set_max_sidebar_width(320.0);

        let toast_overlay = adw::ToastOverlay::new();
        toast_overlay.set_child(Some(&split));
        window.set_content(Some(&toast_overlay));

        Rc::new(Self {
            window: window.clone(),
            state,
            chat_list,
            new_chat_button,
            refresh_button,
            model_dropdown,
            status_label,
            title_widget,
            main_stack,
            messages_box,
            messages_scroll,
            input,
            send_button,
            stop_button,
            context_label,
            context_bar,
            toast_overlay,
            menu_button,
            current_id: RefCell::new(None),
            active: RefCell::new(None),
            suppress_model_change: Cell::new(false),
        })
    }

    fn wire(self: &Rc<Self>) {
        {
            let win = self.clone();
            self.send_button
                .connect_clicked(move |_| win.send_message());
        }
        {
            let win = self.clone();
            self.stop_button
                .connect_clicked(move |_| win.stop_streaming());
        }
        {
            let win = self.clone();
            let controller = gtk::EventControllerKey::new();
            controller.connect_key_pressed(move |_, key, _, modifier| {
                let enter = key == gdk::Key::Return || key == gdk::Key::KP_Enter;
                if enter && !modifier.contains(gdk::ModifierType::SHIFT_MASK) {
                    win.send_message();
                    glib::Propagation::Stop
                } else {
                    glib::Propagation::Proceed
                }
            });
            self.input.add_controller(controller);
        }
        {
            let input = self.input.clone();
            self.input.buffer().connect_changed(move |buffer| {
                let lines = buffer.line_count().clamp(1, 8);
                input.set_height_request(lines * 22 + 16);
            });
        }
        {
            let win = self.clone();
            self.chat_list.connect_row_activated(move |_, row| {
                if let Ok(id) = row.widget_name().parse::<i64>() {
                    win.load_conversation(id);
                }
            });
        }
        {
            let win = self.clone();
            self.new_chat_button
                .connect_clicked(move |_| win.new_chat());
        }
        {
            let win = self.clone();
            self.refresh_button
                .connect_clicked(move |_| win.refresh_models());
        }
        {
            let win = self.clone();
            self.model_dropdown.connect_selected_notify(move |_| {
                win.on_model_changed();
            });
        }

        // Menu + window actions
        let menu = gio::Menu::new();
        let section = gio::Menu::new();
        section.append(Some("New Chat"), Some("win.new-chat"));
        section.append(Some("Preferences"), Some("win.preferences"));
        section.append(Some("About Ollama GUI"), Some("win.about"));
        section.append(Some("Quit"), Some("app.quit"));
        menu.append_section(None, &section);
        self.menu_button.set_menu_model(Some(&menu));

        {
            let action = gio::SimpleAction::new("new-chat", None);
            let win = self.clone();
            action.connect_activate(move |_, _| win.new_chat());
            self.window.add_action(&action);
        }
        {
            let action = gio::SimpleAction::new("preferences", None);
            let win = self.clone();
            action.connect_activate(move |_, _| win.show_settings());
            self.window.add_action(&action);
        }
        {
            let action = gio::SimpleAction::new("about", None);
            let win = self.clone();
            action.connect_activate(move |_, _| win.show_about());
            self.window.add_action(&action);
        }

        if let Some(app) = self.window.application() {
            app.set_accels_for_action("win.new-chat", &["<primary>n"]);
            app.set_accels_for_action("win.preferences", &["<primary>comma"]);
        }
    }

    fn start(self: &Rc<Self>) {
        self.apply_theme();
        self.reload_sidebar();

        let first = self
            .state
            .store
            .borrow()
            .list_conversations()
            .unwrap_or_default()
            .into_iter()
            .next();

        match first {
            Some(conversation) => self.load_conversation(conversation.id),
            None => self.show_empty(),
        }

        self.refresh_models();
        self.input.grab_focus();
    }

    // ----- Models ----------------------------------------------------------

    fn selected_model(&self) -> String {
        let index = self.model_dropdown.selected();
        if index == u32::MAX {
            return String::new();
        }
        self.model_dropdown
            .model()
            .and_downcast::<gtk::StringList>()
            .and_then(|list| list.string(index))
            .map(|value| value.to_string())
            .unwrap_or_default()
    }

    fn select_model_in_dropdown(&self, model: &str) {
        let Some(list) = self
            .model_dropdown
            .model()
            .and_downcast::<gtk::StringList>()
        else {
            return;
        };
        let mut found = None;
        for index in 0..list.n_items() {
            if list.string(index).map(|value| value.to_string()).as_deref() == Some(model) {
                found = Some(index);
                break;
            }
        }
        if let Some(index) = found {
            self.suppress_model_change.set(true);
            self.model_dropdown.set_selected(index);
            self.suppress_model_change.set(false);
        }
    }

    fn refresh_models(self: &Rc<Self>) {
        self.set_status("Connecting…", None);
        let client = self.state.client.borrow().clone();
        let (tx, rx) = async_channel::bounded(1);
        self.state.runtime.spawn(async move {
            let result = client.list_models().await;
            let _ = tx.send(result).await;
        });

        let win = self.clone();
        glib::spawn_future_local(async move {
            match rx.recv().await {
                Ok(Ok(models)) => win.apply_models(models),
                Ok(Err(err)) => win.set_offline(&format!("{err:#}")),
                Err(_) => {}
            }
        });
    }

    fn apply_models(self: &Rc<Self>, models: Vec<ModelSummary>) {
        let models: Vec<ModelSummary> = models
            .into_iter()
            .filter(ModelSummary::is_chat_model)
            .collect();
        let names: Vec<String> = models.iter().map(|model| model.name.clone()).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let list = gtk::StringList::new(&refs);

        let previous = self.selected_model();
        let conversation_model = self.current_id.borrow().and_then(|id| {
            self.state
                .store
                .borrow()
                .conversation_model(id)
                .ok()
                .flatten()
        });
        let default_model = self.state.settings.borrow().default_model.clone();

        self.suppress_model_change.set(true);
        self.model_dropdown.set_model(Some(&list));
        let preferred = conversation_model
            .filter(|model| names.contains(model))
            .or_else(|| (!previous.is_empty() && names.contains(&previous)).then_some(previous))
            .or_else(|| {
                (!default_model.is_empty() && names.contains(&default_model))
                    .then_some(default_model)
            })
            .or_else(|| names.first().cloned());

        if let Some(model) = preferred {
            if let Some(index) = names.iter().position(|name| *name == model) {
                self.model_dropdown.set_selected(index as u32);
            }
        } else {
            self.model_dropdown.set_selected(u32::MAX);
        }
        self.suppress_model_change.set(false);

        *self.state.models.borrow_mut() = models;

        if names.is_empty() {
            self.set_status("Online · no chat models installed", Some(true));
        } else {
            let plural = if names.len() == 1 { "" } else { "s" };
            self.set_status(
                &format!("Online · {} chat model{plural}", names.len()),
                Some(true),
            );
        }

        let model = self.selected_model();
        if !model.is_empty() {
            self.title_widget.set_subtitle(&model);
        }
    }

    fn on_model_changed(self: &Rc<Self>) {
        if self.suppress_model_change.get() {
            return;
        }
        let model = self.selected_model();
        if model.is_empty() {
            return;
        }
        self.title_widget.set_subtitle(&model);
        self.state.settings.borrow_mut().default_model = model.clone();
        self.save_settings();
        if let Some(id) = *self.current_id.borrow() {
            let _ = self.state.store.borrow().set_conversation_model(id, &model);
            self.reload_sidebar();
        }
    }

    fn set_status(&self, text: &str, online: Option<bool>) {
        self.status_label.set_text(text);
        self.status_label.remove_css_class("status-online");
        self.status_label.remove_css_class("status-offline");
        match online {
            Some(true) => {
                self.status_label.remove_css_class("dim-label");
                self.status_label.add_css_class("status-online")
            }
            Some(false) => {
                self.status_label.remove_css_class("dim-label");
                self.status_label.add_css_class("status-offline")
            }
            None => self.status_label.add_css_class("dim-label"),
        }
    }

    fn set_offline(&self, detail: &str) {
        self.set_status("Offline · check Preferences", Some(false));
        self.status_label.set_tooltip_text(Some(detail));
        tracing::warn!("Ollama unavailable: {detail}");
    }

    // ----- Context usage ---------------------------------------------------

    fn update_context_indicator(&self, used: Option<u32>) {
        let total = self.state.settings.borrow().num_ctx.max(1);
        let used_tokens = used.unwrap_or(0);
        let fraction = used_tokens as f64 / total as f64;
        let percent = fraction * 100.0;

        self.context_label.set_text(&format!(
            "Context: {} / {} tokens · {}",
            group_digits(used_tokens),
            group_digits(total),
            format_percent(percent)
        ));
        self.context_bar.set_value(fraction.clamp(0.0, 1.0));
        apply_usage_class(&self.context_label, percent);
        self.context_label.set_tooltip_text(Some(&format!(
            "Estimated context usage: {} of {} tokens ({})",
            group_digits(used_tokens),
            group_digits(total),
            format_percent(percent)
        )));
    }

    fn refresh_context_indicator(&self) {
        let used = self.current_id.borrow().and_then(|id| {
            self.state
                .store
                .borrow()
                .conversation_context_used(id)
                .ok()
                .flatten()
        });
        self.update_context_indicator(used);
    }

    // ----- Conversations ---------------------------------------------------

    fn reload_sidebar(self: &Rc<Self>) {
        while let Some(child) = self.chat_list.first_child() {
            self.chat_list.remove(&child);
        }

        let conversations = self
            .state
            .store
            .borrow()
            .list_conversations()
            .unwrap_or_default();
        let current = *self.current_id.borrow();
        let num_ctx = self.state.settings.borrow().num_ctx.max(1);

        for conversation in conversations {
            let action_row = adw::ActionRow::builder()
                .title(&conversation.title)
                .subtitle(&conversation.model)
                .activatable(true)
                .build();

            let used = conversation.context_used.unwrap_or(0);
            let percent = used as f64 / num_ctx as f64 * 100.0;
            let percent_label = gtk::Label::new(Some(&format_percent(percent)));
            percent_label.set_valign(gtk::Align::Center);
            percent_label.set_width_chars(5);
            percent_label.set_xalign(1.0);
            apply_usage_class(&percent_label, percent);
            percent_label.set_tooltip_text(Some(&format!(
                "Context: {} / {} tokens ({})",
                group_digits(used),
                group_digits(num_ctx),
                format_percent(percent)
            )));
            action_row.add_suffix(&percent_label);

            let delete_button = gtk::Button::from_icon_name("user-trash-symbolic");
            delete_button.add_css_class("flat");
            delete_button.set_tooltip_text(Some("Delete chat"));
            delete_button.set_valign(gtk::Align::Center);
            action_row.add_suffix(&delete_button);
            action_row.set_widget_name(&conversation.id.to_string());

            {
                let win = self.clone();
                let id = conversation.id;
                delete_button.connect_clicked(move |_| {
                    if let Err(err) = win.state.store.borrow().delete_conversation(id) {
                        win.toast(&format!("Could not delete chat: {err:#}"));
                        return;
                    }
                    if *win.current_id.borrow() == Some(id) {
                        win.new_chat();
                    }
                    win.reload_sidebar();
                });
            }

            self.chat_list.append(&action_row);

            if current == Some(conversation.id) {
                self.chat_list.select_row(Some(&action_row));
            }
        }
    }

    fn load_conversation(self: &Rc<Self>, id: i64) {
        self.stop_streaming();
        *self.current_id.borrow_mut() = Some(id);

        let (title, model, messages) = {
            let store = self.state.store.borrow();
            (
                store
                    .conversation_title(id)
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| "Chat".to_string()),
                store
                    .conversation_model(id)
                    .ok()
                    .flatten()
                    .unwrap_or_default(),
                store.messages(id).unwrap_or_default(),
            )
        };

        self.clear_messages();
        for message in &messages {
            self.add_message_widget(&message.role, &message.content);
        }

        self.title_widget.set_title(&title);
        if model.is_empty() {
            self.title_widget.set_subtitle("No model selected");
        } else {
            self.title_widget.set_subtitle(&model);
        }
        self.select_model_in_dropdown(&model);

        if messages.is_empty() {
            self.main_stack.set_visible_child_name("empty");
        } else {
            self.main_stack.set_visible_child_name("chat");
            self.scroll_to_bottom();
        }
        self.refresh_context_indicator();
    }

    fn show_empty(&self) {
        self.clear_messages();
        self.main_stack.set_visible_child_name("empty");
        self.title_widget.set_title("New Chat");
        let model = self.selected_model();
        if model.is_empty() {
            self.title_widget.set_subtitle("No model selected");
        } else {
            self.title_widget.set_subtitle(&model);
        }
        self.refresh_context_indicator();
    }

    fn new_chat(self: &Rc<Self>) {
        self.stop_streaming();
        *self.current_id.borrow_mut() = None;
        self.chat_list.unselect_all();
        self.show_empty();
        self.input.grab_focus();
    }

    // ----- Messages --------------------------------------------------------

    fn clear_messages(&self) {
        while let Some(child) = self.messages_box.first_child() {
            self.messages_box.remove(&child);
        }
    }

    fn add_message_widget(&self, role: &str, text: &str) {
        if role == "user" {
            self.add_user_message(text);
        } else {
            let (_row, _bubble, view) = self.add_assistant_bubble();
            super::markdown::render(&view, text);
        }
    }

    fn add_user_message(&self, text: &str) {
        let label = gtk::Label::new(Some(text));
        label.set_wrap(true);
        label.set_wrap_mode(gtk::pango::WrapMode::WordChar);
        label.set_max_width_chars(60);
        label.set_selectable(true);
        label.set_xalign(0.0);

        let bubble = gtk::Box::new(gtk::Orientation::Vertical, 0);
        bubble.add_css_class("user-bubble");
        bubble.set_halign(gtk::Align::End);
        bubble.append(&label);

        let row = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        row.set_halign(gtk::Align::Fill);
        row.append(&bubble);
        self.messages_box.append(&row);
    }

    fn add_assistant_bubble(&self) -> (gtk::Box, gtk::Box, gtk::TextView) {
        let view = gtk::TextView::builder()
            .editable(false)
            .cursor_visible(false)
            .wrap_mode(gtk::WrapMode::WordChar)
            .build();
        view.add_css_class("message-text");
        view.set_can_focus(true);

        let bubble = gtk::Box::new(gtk::Orientation::Vertical, 4);
        bubble.add_css_class("assistant-bubble");
        bubble.set_halign(gtk::Align::Fill);
        bubble.set_hexpand(true);
        bubble.append(&view);

        let row = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        row.set_halign(gtk::Align::Fill);
        row.append(&bubble);
        self.messages_box.append(&row);

        (row, bubble, view)
    }

    fn near_bottom(&self) -> bool {
        let adjustment = self.messages_scroll.vadjustment();
        adjustment.value() + adjustment.page_size() >= adjustment.upper() - 80.0
    }

    fn scroll_to_bottom(&self) {
        let adjustment = self.messages_scroll.vadjustment();
        glib::idle_add_local_once(move || {
            adjustment.set_value(adjustment.upper() - adjustment.page_size());
        });
    }

    fn send_message(self: &Rc<Self>) {
        if self.active.borrow().is_some() {
            return;
        }

        let buffer = self.input.buffer();
        let text = buffer
            .text(&buffer.start_iter(), &buffer.end_iter(), true)
            .trim()
            .to_string();
        if text.is_empty() {
            return;
        }

        let model = self.selected_model();
        if model.is_empty() {
            self.toast("No chat model selected. Pull one with `ollama pull`, then refresh.");
            return;
        }

        let existing_id = *self.current_id.borrow();
        let conversation_id = match existing_id {
            Some(id) => {
                let _ = self.state.store.borrow().set_conversation_model(id, &model);
                id
            }
            None => {
                let title = title_from(&text);
                let id = match self
                    .state
                    .store
                    .borrow()
                    .create_conversation(&model, &title)
                {
                    Ok(id) => id,
                    Err(err) => {
                        self.toast(&format!("Could not create chat: {err:#}"));
                        return;
                    }
                };
                *self.current_id.borrow_mut() = Some(id);
                self.title_widget.set_title(&title);
                id
            }
        };

        if let Err(err) =
            self.state
                .store
                .borrow()
                .add_message(conversation_id, "user", &text, None)
        {
            self.toast(&format!("Could not save message: {err:#}"));
            return;
        }

        self.add_user_message(&text);
        self.main_stack.set_visible_child_name("chat");

        buffer.set_text("");
        self.reload_sidebar();

        let history = match self.state.store.borrow().messages(conversation_id) {
            Ok(messages) => messages,
            Err(err) => {
                self.toast(&format!("Could not load chat: {err:#}"));
                return;
            }
        };

        let system_prompt = self
            .state
            .settings
            .borrow()
            .system_prompt
            .trim()
            .to_string();
        let mut messages = Vec::new();
        if !system_prompt.is_empty() {
            messages.push(ChatMessage::new("system", system_prompt));
        }
        messages.extend(
            history
                .into_iter()
                .map(|message| ChatMessage::new(&message.role, message.content)),
        );

        let (temperature, num_ctx) = {
            let settings = self.state.settings.borrow();
            (settings.temperature, settings.num_ctx)
        };

        let request = ChatRequest {
            model,
            messages,
            stream: true,
            options: Some(ChatOptions {
                temperature: Some(temperature),
                num_ctx: Some(num_ctx),
                ..Default::default()
            }),
            keep_alive: Some("5m".to_string()),
        };

        let (row, bubble, assistant_view) = self.add_assistant_bubble();
        let assistant_buffer = assistant_view.buffer();
        let content = Rc::new(RefCell::new(String::new()));
        let stats: Rc<RefCell<Option<StreamStats>>> = Rc::new(RefCell::new(None));
        let (tx, rx) = async_channel::unbounded();

        let client = self.state.client.borrow().clone();
        let abort = api::spawn_chat_stream(&self.state.runtime, client, request, tx);

        *self.active.borrow_mut() = Some(ActiveStream {
            abort,
            row,
            bubble,
            view: assistant_view,
            content: content.clone(),
            conv_id: conversation_id,
            stats: stats.clone(),
        });

        self.set_streaming_ui(true);
        self.scroll_to_bottom();

        let win = self.clone();
        glib::spawn_future_local(async move {
            let mut had_error = false;
            let mut final_stats = None;
            while let Ok(event) = rx.recv().await {
                match event {
                    StreamEvent::Delta(delta) => {
                        let follow = win.near_bottom();
                        assistant_buffer.insert(&mut assistant_buffer.end_iter(), &delta);
                        content.borrow_mut().push_str(&delta);
                        if follow {
                            win.scroll_to_bottom();
                        }
                    }
                    StreamEvent::Done(stream_stats) => {
                        final_stats = Some(stream_stats);
                    }
                    StreamEvent::Error(message) => {
                        had_error = true;
                        win.toast(&message);
                    }
                }
            }

            if let Some(stream_stats) = final_stats {
                *stats.borrow_mut() = Some(stream_stats);
            }
            win.finish_stream(had_error);
        });
    }

    fn finish_stream(self: &Rc<Self>, had_error: bool) {
        let Some(active) = self.active.borrow_mut().take() else {
            return;
        };

        let content = active.content.borrow().clone();
        let stats = active.stats.borrow().clone();

        if !content.trim().is_empty() {
            let conversation_exists = self
                .state
                .store
                .borrow()
                .conversation_model(active.conv_id)
                .ok()
                .flatten()
                .is_some();
            if conversation_exists {
                if let Err(err) = self.state.store.borrow().add_message(
                    active.conv_id,
                    "assistant",
                    &content,
                    stats.as_ref(),
                ) {
                    self.toast(&format!("Could not save reply: {err:#}"));
                }
            }
            if active.row.parent().is_some() {
                super::markdown::render(&active.view, &content);
            }
        } else if active.row.parent().is_some() {
            self.messages_box.remove(&active.row);
        }

        tracing::info!(
            "stream finished for conversation {} ({} chars)",
            active.conv_id,
            content.len()
        );

        if content.trim().is_empty() && had_error {
            self.toast("Generation failed. Check the Ollama server.");
        } else if content.trim().is_empty() {
            self.toast("The model returned an empty response.");
        }

        if let Some(stats) = stats {
            if let Some(tokens_per_second) = stats.tokens_per_second() {
                let label = gtk::Label::new(Some(&format!("{tokens_per_second:.1} tok/s")));
                label.add_css_class("dim-label");
                label.set_halign(gtk::Align::Start);
                label.set_margin_start(6);
                active.bubble.append(&label);
            }
        }

        self.set_streaming_ui(false);
        self.reload_sidebar();
        self.refresh_context_indicator();
        self.scroll_to_bottom();
    }

    fn stop_streaming(&self) {
        if let Some(active) = self.active.borrow().as_ref() {
            active.abort.abort();
        }
        self.set_streaming_ui(false);
    }

    fn set_streaming_ui(&self, streaming: bool) {
        self.send_button.set_sensitive(!streaming);
        self.stop_button.set_visible(streaming);
    }

    // ----- Settings / theme / misc ----------------------------------------

    fn save_settings(&self) {
        if let Err(err) = self.state.settings.borrow().save() {
            self.toast(&format!("Could not save settings: {err:#}"));
        }
    }

    fn apply_theme(&self) {
        let theme = self.state.settings.borrow().theme.clone();
        let scheme = match theme.as_str() {
            "light" => adw::ColorScheme::ForceLight,
            "dark" => adw::ColorScheme::ForceDark,
            _ => adw::ColorScheme::Default,
        };
        adw::StyleManager::default().set_color_scheme(scheme);
    }

    fn show_settings(self: &Rc<Self>) {
        let page = adw::PreferencesPage::new();

        let connection = adw::PreferencesGroup::new();
        connection.set_title("Connection");

        let url_row = adw::EntryRow::new();
        url_row.set_title("Server URL");
        url_row.set_text(&self.state.settings.borrow().base_url);
        connection.add(&url_row);

        let key_row = adw::PasswordEntryRow::new();
        key_row.set_title("API key (optional)");
        key_row.set_text(&self.state.settings.borrow().api_key);
        connection.add(&key_row);

        page.add(&connection);

        let generation = adw::PreferencesGroup::new();
        generation.set_title("Generation");

        let system_row = adw::EntryRow::new();
        system_row.set_title("System prompt");
        system_row.set_text(&self.state.settings.borrow().system_prompt);
        generation.add(&system_row);

        let temperature_row = adw::SpinRow::new(
            Some(&gtk::Adjustment::new(
                self.state.settings.borrow().temperature,
                0.0,
                2.0,
                0.05,
                0.1,
                0.0,
            )),
            0.05,
            2,
        );
        temperature_row.set_title("Temperature");
        temperature_row.set_subtitle("Higher values produce more varied replies");
        generation.add(&temperature_row);

        let context_row = adw::SpinRow::new(
            Some(&gtk::Adjustment::new(
                self.state.settings.borrow().num_ctx as f64,
                512.0,
                262_144.0,
                512.0,
                4096.0,
                0.0,
            )),
            1024.0,
            0,
        );
        context_row.set_title("Context size");
        context_row.set_subtitle("num_ctx passed to Ollama");
        generation.add(&context_row);

        page.add(&generation);

        let appearance = adw::PreferencesGroup::new();
        appearance.set_title("Appearance");

        let theme_row = adw::ComboRow::new();
        theme_row.set_title("Theme");
        let themes = gtk::StringList::new(&["Follow system", "Light", "Dark"]);
        theme_row.set_model(Some(&themes));
        theme_row.set_selected(match self.state.settings.borrow().theme.as_str() {
            "light" => 1,
            "dark" => 2,
            _ => 0,
        });
        appearance.add(&theme_row);

        page.add(&appearance);

        {
            let win = self.clone();
            url_row.connect_changed(move |row| {
                win.state.settings.borrow_mut().base_url = row.text().to_string();
                win.save_settings();
                win.state.rebuild_client();
            });
        }
        {
            let win = self.clone();
            key_row.connect_changed(move |row| {
                win.state.settings.borrow_mut().api_key = row.text().to_string();
                win.save_settings();
                win.state.rebuild_client();
            });
        }
        {
            let win = self.clone();
            system_row.connect_changed(move |row| {
                win.state.settings.borrow_mut().system_prompt = row.text().to_string();
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            temperature_row.connect_value_notify(move |row| {
                win.state.settings.borrow_mut().temperature = row.value();
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            context_row.connect_value_notify(move |row| {
                win.state.settings.borrow_mut().num_ctx = row.value() as u32;
                win.save_settings();
                win.refresh_context_indicator();
                win.reload_sidebar();
            });
        }
        {
            let win = self.clone();
            theme_row.connect_selected_notify(move |row| {
                let theme = match row.selected() {
                    1 => "light",
                    2 => "dark",
                    _ => "system",
                };
                win.state.settings.borrow_mut().theme = theme.to_string();
                win.save_settings();
                win.apply_theme();
            });
        }

        let dialog = adw::PreferencesDialog::builder()
            .title("Preferences")
            .build();
        dialog.add(&page);
        dialog.present(Some(&self.window));
    }

    fn show_about(&self) {
        let about = adw::AboutDialog::builder()
            .application_name("Ollama GUI")
            .application_icon("io.github.ltux.OllamaGui")
            .developer_name("ltux")
            .version(env!("CARGO_PKG_VERSION"))
            .comments("A native GTK4/libadwaita client for Ollama.")
            .build();
        about.present(Some(&self.window));
    }

    fn toast(&self, message: &str) {
        self.toast_overlay.add_toast(adw::Toast::new(message));
    }
}

fn title_from(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or("New Chat").trim();
    let mut title: String = first_line.chars().take(48).collect();
    if first_line.chars().count() > 48 {
        title.push('…');
    }
    if title.is_empty() {
        "New Chat".to_string()
    } else {
        title
    }
}

fn group_digits(value: u32) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    for (index, ch) in text.chars().enumerate() {
        if index > 0 && (text.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn format_percent(percent: f64) -> String {
    if percent > 0.0 && percent < 10.0 {
        format!("{percent:.1}%")
    } else {
        format!("{percent:.0}%")
    }
}

fn apply_usage_class(label: &gtk::Label, percent: f64) {
    label.remove_css_class("dim-label");
    label.remove_css_class("context-warning");
    label.remove_css_class("context-high");
    if percent >= 90.0 {
        label.add_css_class("context-high");
    } else if percent >= 75.0 {
        label.add_css_class("context-warning");
    } else {
        label.add_css_class("dim-label");
    }
}

#[cfg(test)]
mod tests {
    use super::{format_percent, group_digits};

    #[test]
    fn groups_thousands() {
        assert_eq!(group_digits(0), "0");
        assert_eq!(group_digits(999), "999");
        assert_eq!(group_digits(8_192), "8,192");
        assert_eq!(group_digits(1_234_567), "1,234,567");
    }

    #[test]
    fn formats_small_percentages_with_one_decimal() {
        assert_eq!(format_percent(0.0), "0%");
        assert_eq!(format_percent(0.28), "0.3%");
        assert_eq!(format_percent(9.9), "9.9%");
        assert_eq!(format_percent(15.4), "15%");
    }
}
