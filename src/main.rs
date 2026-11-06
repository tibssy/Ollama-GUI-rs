mod api;
mod config;
mod store;
mod ui;

use gtk::gio;
use gtk::prelude::*;
use libadwaita as adw;
use libadwaita::prelude::*;

fn main() -> gtk::glib::ExitCode {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let app = adw::Application::builder()
        .application_id("io.github.ltux.OllamaGui")
        .build();

    app.connect_startup(|app| {
        let provider = gtk::CssProvider::new();
        provider.load_from_string(include_str!("../resources/style.css"));
        if let Some(display) = gtk::gdk::Display::default() {
            gtk::style_context_add_provider_for_display(
                &display,
                &provider,
                gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
            );
        }

        let quit = gio::SimpleAction::new("quit", None);
        let app_for_quit = app.clone();
        quit.connect_activate(move |_, _| app_for_quit.quit());
        app.add_action(&quit);
        app.set_accels_for_action("app.quit", &["<primary>q"]);
    });

    app.connect_activate(|app| match ui::build(app) {
        Ok(window) => window.present(),
        Err(err) => {
            tracing::error!("failed to start: {err:#}");
            let window = adw::ApplicationWindow::builder()
                .application(app)
                .title("Ollama GUI")
                .default_width(520)
                .default_height(240)
                .build();
            let page = adw::StatusPage::builder()
                .title("Could not start Ollama GUI")
                .description(format!("{err:#}"))
                .build();
            window.set_content(Some(&page));
            window.present();
        }
    });

    app.run()
}
