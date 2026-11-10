use async_channel::Sender;

/// Commands sent from the status-notifier thread to the GTK main loop.
#[derive(Debug, Clone)]
pub enum TrayCommand {
    ToggleWindow(Option<String>),
    ToggleLive,
    ToggleWake,
    Quit,
}

pub struct StatusTray {
    pub commands: Sender<TrayCommand>,
    pub live: bool,
    pub live_available: bool,
    /// True when wake word detection was enabled in settings.
    pub wake_visible: bool,
    /// True while the microphone is listening for the wake word.
    pub wake_listening: bool,
    pub wake_word: String,
}

/// The activation token may arrive a moment after the tray event.
fn activation_token() -> Option<String> {
    ksni::current_xdg_activation_token().or_else(|| {
        std::thread::sleep(std::time::Duration::from_millis(2));
        ksni::current_xdg_activation_token()
    })
}

impl ksni::Tray for StatusTray {
    fn id(&self) -> String {
        "io.github.ltux.OllamaGui".to_string()
    }

    fn title(&self) -> String {
        if self.live {
            "Ollama GUI — Live conversation".to_string()
        } else {
            "Ollama GUI".to_string()
        }
    }

    fn icon_name(&self) -> String {
        if self.live {
            "io.github.ltux.OllamaGui-live".to_string()
        } else if self.wake_listening {
            "io.github.ltux.OllamaGui-wake".to_string()
        } else {
            "io.github.ltux.OllamaGui".to_string()
        }
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip {
            title: "Ollama GUI".to_string(),
            description: if self.live {
                "Live voice conversation is active".to_string()
            } else if self.wake_visible {
                format!("Listening for \u{201c}{}\u{201d}", self.wake_word)
            } else {
                "Ready".to_string()
            },
            ..Default::default()
        }
    }

    fn activate(&mut self, _x: i32, _y: i32) {
        let token = activation_token();
        let _ = self.commands.try_send(TrayCommand::ToggleWindow(token));
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::{CheckmarkItem, MenuItem, StandardItem};

        vec![
            StandardItem {
                label: "Show/Hide Window".to_string(),
                activate: Box::new(|tray: &mut StatusTray| {
                    let token = activation_token();
                    let _ = tray.commands.try_send(TrayCommand::ToggleWindow(token));
                }),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: if self.live {
                    "Stop Live Conversation".to_string()
                } else {
                    "Start Live Conversation".to_string()
                },
                enabled: self.live || self.live_available,
                activate: Box::new(|tray: &mut StatusTray| {
                    let _ = tray.commands.try_send(TrayCommand::ToggleLive);
                }),
                ..Default::default()
            }
            .into(),
            CheckmarkItem {
                label: format!("Wake word: {}", self.wake_word),
                visible: self.wake_visible,
                enabled: self.wake_visible && !self.live,
                checked: self.wake_listening,
                activate: Box::new(|tray: &mut StatusTray| {
                    let _ = tray.commands.try_send(TrayCommand::ToggleWake);
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Quit".to_string(),
                icon_name: "application-exit".to_string(),
                activate: Box::new(|tray: &mut StatusTray| {
                    let _ = tray.commands.try_send(TrayCommand::Quit);
                }),
                ..Default::default()
            }
            .into(),
        ]
    }
}
