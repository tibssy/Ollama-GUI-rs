use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

use crate::config;

const FILE_NAME: &str = "io.github.ltux.OllamaGui.desktop";

/// Path of the XDG autostart entry managed by the app.
pub fn desktop_path() -> PathBuf {
    desktop_path_in(&config::config_home())
}

fn desktop_path_in(config_home: &Path) -> PathBuf {
    config_home.join("autostart").join(FILE_NAME)
}

/// True when the app is registered to start with the graphical session.
pub fn is_enabled() -> bool {
    desktop_path().is_file()
}

/// Write or remove the autostart entry. The file is the source of truth, so
/// the settings switch always reflects the actual desktop state.
pub fn set_enabled(enabled: bool) -> Result<()> {
    write_entry_at(&desktop_path(), enabled)
}

fn write_entry_at(path: &Path, enabled: bool) -> Result<()> {
    if enabled {
        let exe = std::env::current_exe().context("locating the Ollama GUI executable")?;
        let contents = desktop_entry(&exe);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(path, contents).with_context(|| format!("writing {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(path)?.permissions();
            perms.set_mode(0o644);
            fs::set_permissions(path, perms)?;
        }
    } else if path.is_file() {
        fs::remove_file(path).with_context(|| format!("removing {}", path.display()))?;
    }
    Ok(())
}

fn desktop_entry(exe: &Path) -> String {
    format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=Ollama GUI\n\
         Comment=Start Ollama GUI hidden in the system tray\n\
         Exec={} --autostart\n\
         Icon=io.github.ltux.OllamaGui\n\
         Terminal=false\n\
         NoDisplay=true\n\
         StartupNotify=false\n\
         X-GNOME-Autostart-enabled=true\n",
        quote_exec_arg(&exe.to_string_lossy())
    )
}

/// Quote an Exec argument using the desktop entry rules. Always quoting keeps
/// paths with spaces and shell metacharacters safe.
fn quote_exec_arg(arg: &str) -> String {
    let mut quoted = String::with_capacity(arg.len() + 2);
    quoted.push('"');
    for ch in arg.chars() {
        if matches!(ch, '\\' | '"' | '`' | '$') {
            quoted.push('\\');
        }
        quoted.push(ch);
    }
    quoted.push('"');
    quoted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_starts_the_app_with_the_autostart_flag() {
        let entry = desktop_entry(Path::new("/home/me/ollama-gui"));
        assert!(entry.contains("Exec=\"/home/me/ollama-gui\" --autostart"));
        assert!(entry.contains("X-GNOME-Autostart-enabled=true"));
        assert!(!entry.contains("X-GNOME-Autostart-Phase"));
    }

    #[test]
    fn exec_arguments_are_quoted_for_desktop_entries() {
        assert_eq!(
            quote_exec_arg("/home/me/ollama-gui"),
            "\"/home/me/ollama-gui\""
        );
        assert_eq!(
            quote_exec_arg("/home/me/My Apps/ollama-gui"),
            "\"/home/me/My Apps/ollama-gui\""
        );
        assert_eq!(
            quote_exec_arg("/home/me/a\"b$c`d\\e"),
            "\"/home/me/a\\\"b\\$c\\`d\\\\e\""
        );
    }

    #[test]
    fn entry_is_written_and_removed() {
        let root =
            std::env::temp_dir().join(format!("ollama-gui-autostart-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let path = desktop_path_in(&root);

        write_entry_at(&path, true).expect("write entry");
        assert!(path.is_file());
        let contents = fs::read_to_string(&path).expect("read entry");
        assert!(contents.contains("--autostart"));

        write_entry_at(&path, false).expect("remove entry");
        assert!(!path.exists());

        let _ = fs::remove_dir_all(&root);
    }
}
