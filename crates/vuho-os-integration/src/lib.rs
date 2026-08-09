//! macOS OS integration: clipboard, text injection, language detection, hotkeys.
//!
//! Native macOS APIs via the `objc2` family (`NSPasteboard`, `CGEvent`, TIS,
//! Security.framework). No Swift FFI, no `arboard` dependency.

// ── Submodules (sys must come first — other modules depend on it) ─────────

mod clipboard;
mod hotkey;
mod inject;
mod language;
mod sys;

// ── Re-exports ────────────────────────────────────────────────────────────

pub use clipboard::copy_to_clipboard;
pub use hotkey::{HotkeyConfig, HotkeyListener};
pub use inject::inject_text;
pub use language::{
    cached_input_language, install_language_watcher, map_bcp47_to_whisper, mapped_languages,
    LanguageDetector,
};

/// Trigger the native macOS Accessibility permission prompt and add this process
/// to the Accessibility list (ADR-012). Returns whether access is currently
/// granted. The process must be **relaunched** after granting before the
/// `CapsLock` hotkey's event tap can bind.
#[must_use]
pub fn prompt_accessibility_trust() -> bool {
    sys::prompt_accessibility_trust()
}

/// Query whether the current process has Accessibility permission, without
/// prompting (ADR-016 preflight permission gate).
#[must_use]
pub fn accessibility_trusted() -> bool {
    sys::is_accessibility_trusted()
}

/// Query whether the current process has Input Monitoring permission,
/// without prompting (ADR-016 preflight permission gate).
#[must_use]
pub fn input_monitoring_trusted() -> bool {
    sys::is_input_monitoring_trusted()
}

/// The tri-state result of an Input Monitoring access check — re-exported so
/// callers outside this crate (the permission gate) can distinguish "never
/// asked" (promptable) from an explicit denial, which
/// [`input_monitoring_trusted`]'s collapsed bool cannot express.
pub use sys::InputMonitoringAccess;

/// Query the current process's Input Monitoring access as the tri-state
/// `IOHIDCheckAccess` actually reports it, without prompting (ADR-016
/// preflight permission gate's denied-state handling).
#[must_use]
pub fn input_monitoring_access() -> InputMonitoringAccess {
    sys::input_monitoring_access()
}

/// Trigger the native Input Monitoring permission prompt. Fire-and-forget —
/// does not wait for the user's answer (ADR-016 preflight permission gate).
pub fn request_input_monitoring_access() {
    sys::request_input_monitoring_access();
}

// ── Error type ────────────────────────────────────────────────────────────

/// Error type for OS integration operations.
#[derive(thiserror::Error, Debug)]
pub enum OsError {
    /// Writing to the system clipboard (`NSPasteboard`) failed.
    #[error("clipboard write failed")]
    ClipboardWrite,
    /// Synthesizing the ⌘V keystroke via `CGEvent` failed.
    #[error("text injection failed")]
    InjectionFailed,
    /// Reading the active TIS keyboard input source failed.
    #[error("language detection failed")]
    LanguageDetection,
    /// Creating or enabling the global hotkey `CGEventTap` failed.
    #[error("hotkey registration failed")]
    Hotkey,
    /// [`HotkeyListener::start`] was called on a listener that is already
    /// running. Call [`HotkeyListener::stop`] first — a second, unguarded
    /// `start()` would leak the first tap thread unstoppably (its
    /// `stopped` handle would be overwritten, so `stop()` could never
    /// signal it again).
    #[error("hotkey listener is already running; call stop() first")]
    HotkeyAlreadyRunning,
    /// macOS Secure Input was active, so the ⌘V keystroke could not be
    /// delivered — the text is still on the clipboard for a manual paste.
    #[error("secure input active; text left on clipboard")]
    SecureInputActive,
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_detector_returns_valid_code() {
        // TIS is main-thread-only (uncatchable SIGTRAP off-main — this test
        // used to abort the whole release-mode test binary). Test threads are
        // never the main thread, so exercise the marker-gated path only when
        // a marker exists; otherwise this verifies the compile-time contract.
        let Some(mtm) = objc2::MainThreadMarker::new() else {
            eprintln!("skipping: not on the main thread (TIS is main-thread-only)");
            return;
        };
        if let Ok(lang) = LanguageDetector::current_input_language(mtm) {
            assert_eq!(lang.len(), 2);
        }
    }

    #[test]
    fn clipboard_copy_smoke() {
        // May fail in headless/env without display, so just ensure it doesn't panic.
        let _ = copy_to_clipboard("test");
    }

    #[test]
    fn hotkey_listener_starts_and_stops() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut listener = HotkeyListener::new();
        // May fail if Accessibility is not granted (CI).
        let _ = listener.start(&tx, HotkeyConfig::default());
        listener.stop();
    }
}
