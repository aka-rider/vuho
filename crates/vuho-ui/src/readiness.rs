//! Permission data model (ADR-016) — the three TCC grants Vuho needs, their
//! live tri-state [`Access`], and the pure preflight check
//! [`missing_permissions`], plus a couple of small formatting helpers shared
//! by [`crate::app_status::StatusModel`] and [`crate::settings_tab::SettingsTab`].
//!
//! The window that used to live in this module — the ADR-016 permission
//! gate / ADR-020 readiness window — is gone (WP6, ARCHITECTURE.md ADR-021):
//! its two jobs are now both the panel's Settings tab
//! ([`crate::settings_tab::SettingsTab`], which reads [`Permission::ALL`]/
//! [`Access`]/[`missing_permissions`] directly) opened on launch via
//! `crate::panel::show` when [`missing_permissions`] is non-empty. What
//! remains here is exactly the data model + pure helpers those two callers
//! need — no window, no polling loop (`crate::panel`'s own
//! `start_permissions_poll` replaces it), no `AppKit` window construction.

use vuho_domain::ModelStatus;
use vuho_os_integration::InputMonitoringAccess;
use vuho_stt_engine::MicAuthStatus;

use crate::permissions::{
    ACCESSIBILITY_SETTINGS_URL, INPUT_MONITORING_SETTINGS_URL, MICROPHONE_SETTINGS_URL,
};

/// Bytes per megabyte for the model-size display ("474 MB") — decimal MB,
/// matching the figure used throughout `ARCHITECTURE.md`/`CLAUDE.md`, not a
/// binary MiB.
const BYTES_PER_MB: u64 = 1_000_000;

// ── Access (tri-state OS grant status — CONSTITUTION rule 2: model it as
//    data, don't infer "denied" from click-then-poll timing) ──────────────

/// The tri-state status of one permission, as the OS actually reports it —
/// not collapsed to a bool. `Promptable` means "never asked, the native
/// one-click prompt will work"; `Denied` means the user (or an MDM policy)
/// already answered no, so re-firing the same prompt is a silent no-op and
/// the only way forward is System Settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    Granted,
    Promptable,
    Denied,
}

/// Map microphone `MicAuthStatus` to `Access`. Pure — unit-tested without
/// any macOS/TCC state (see `tests` below).
fn mic_access(status: MicAuthStatus) -> Access {
    match status {
        MicAuthStatus::Authorized => Access::Granted,
        MicAuthStatus::NotDetermined => Access::Promptable,
        MicAuthStatus::Denied | MicAuthStatus::Restricted => Access::Denied,
    }
}

/// Map Input Monitoring's `IOHIDCheckAccess` tri-state to `Access`. Pure.
fn input_monitoring_to_access(access: InputMonitoringAccess) -> Access {
    match access {
        InputMonitoringAccess::Granted => Access::Granted,
        InputMonitoringAccess::Unknown => Access::Promptable,
        InputMonitoringAccess::Denied => Access::Denied,
    }
}

/// Map Accessibility's `AXIsProcessTrusted` bool to `Access`.
///
/// **OS limitation, not a shortcut:** unlike Microphone (`AVFoundation`) and
/// Input Monitoring (`IOHIDCheckAccess`), the Accessibility API has no
/// three-state "not yet asked" vs "explicitly denied" distinction —
/// `AXIsProcessTrusted` is a plain bool. So this can only ever report
/// `Granted` or `Promptable`, never `Denied`. The Settings tab's "Allow
/// Accessibility" button therefore always re-fires
/// `AXIsProcessTrustedWithOptions`, whose *own* native dialog (when the
/// grant was previously denied) already includes an "Open System
/// Settings…" button — the OS itself, not this app, handles that case for
/// Accessibility.
fn accessibility_access(trusted: bool) -> Access {
    if trusted {
        Access::Granted
    } else {
        Access::Promptable
    }
}

// ── Permission (the one data-driven definition — CONSTITUTION rule 26) ────

/// One of the three TCC grants Vuho needs, documented in `README.md`/
/// `CLAUDE.md`'s testing notes (`tccutil reset Microphone|Accessibility|InputMonitoring`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Permission {
    Accessibility,
    InputMonitoring,
    Microphone,
}

impl Permission {
    /// Every permission the gate checks, in the order shown to the user.
    pub(crate) const ALL: [Permission; 3] = [
        Permission::Accessibility,
        Permission::InputMonitoring,
        Permission::Microphone,
    ];

    /// Pure (non-prompting) tri-state check — safe to call on every poll tick.
    pub(crate) fn access(self) -> Access {
        match self {
            Permission::Accessibility => {
                accessibility_access(vuho_os_integration::accessibility_trusted())
            }
            Permission::InputMonitoring => {
                input_monitoring_to_access(vuho_os_integration::input_monitoring_access())
            }
            Permission::Microphone => mic_access(vuho_stt_engine::mic_permission_status()),
        }
    }

    /// This permission's System Settings deep-link, for the `Access::Denied`
    /// "Open System Settings" button. One source of truth for every URL —
    /// `permissions.rs` (CONSTITUTION rule 26).
    pub(crate) fn settings_url(self) -> &'static str {
        match self {
            Permission::Accessibility => ACCESSIBILITY_SETTINGS_URL,
            Permission::InputMonitoring => INPUT_MONITORING_SETTINGS_URL,
            Permission::Microphone => MICROPHONE_SETTINGS_URL,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Permission::Accessibility => "Accessibility",
            Permission::InputMonitoring => "Input Monitoring",
            Permission::Microphone => "Microphone",
        }
    }

    pub(crate) fn description(self) -> &'static str {
        match self {
            Permission::Accessibility => {
                "Lets Vuho listen for the global CapsLock dictation hotkey."
            }
            Permission::InputMonitoring => {
                "Also required by the hotkey listener to receive keyboard events."
            }
            Permission::Microphone => "Lets Vuho capture your voice to transcribe.",
        }
    }

    /// Trigger this permission's native prompt. Fire-and-forget for all
    /// three: none of the underlying calls wait for the user's answer, so
    /// the panel's permissions poll (`crate::panel::start_permissions_poll`)
    /// is what actually observes the grant landing.
    pub(crate) fn request(self) {
        match self {
            Permission::Accessibility => {
                let _ = vuho_os_integration::prompt_accessibility_trust();
            }
            Permission::InputMonitoring => vuho_os_integration::request_input_monitoring_access(),
            Permission::Microphone => {
                let _ = vuho_stt_engine::request_mic_permission();
            }
        }
    }
}

/// The preflight check: every currently-missing permission with its live
/// [`Access`], in [`Permission::ALL`] order. Side-effect-free — safe to call
/// before any other startup work, and repeatedly from the panel's
/// permissions poll.
///
/// Carries `Access` alongside each `Permission` (not just the permission
/// itself) so this one call is the **entire** derivation `StatusModel::
/// permissions_missing` is ever written from (F6) — `settings_tab.rs`'s
/// permission rows then render purely from that stored data instead of each
/// calling [`Permission::access`] again at render time, which raced the
/// poll's own 500 ms tick (the collapsed header could disagree with the
/// rows for up to that long).
#[must_use]
pub(crate) fn missing_permissions() -> Vec<(Permission, Access)> {
    Permission::ALL
        .into_iter()
        .map(|permission| (permission, permission.access()))
        .filter(|(_, access)| *access != Access::Granted)
        .collect()
}

// ── Speech-model formatting helpers ────────────────────────────────────────

/// Human-readable subtitle for a model-status row
/// (`crate::settings_tab::SettingsTab`'s Speech Model section).
pub(crate) fn model_status_text(status: &ModelStatus) -> String {
    match status {
        ModelStatus::Missing { total_bytes } => {
            format!("{} · not yet downloaded", format_mb(*total_bytes))
        }
        ModelStatus::Downloading {
            received_bytes,
            total_bytes,
        } => format!(
            "Downloading… {} of {}",
            format_mb(*received_bytes),
            format_mb(*total_bytes)
        ),
        ModelStatus::Verifying => "Verifying…".to_owned(),
        ModelStatus::Ready => "Ready".to_owned(),
        ModelStatus::Failed { message } => message.clone(),
    }
}

/// Format a byte count as a whole number of megabytes (decimal, "474 MB") —
/// readable, unlike the raw byte count `models.lock.json` stores.
pub(crate) fn format_mb(bytes: u64) -> String {
    format!("{} MB", (bytes + BYTES_PER_MB / 2) / BYTES_PER_MB)
}

/// Re-exec the current binary and exit this process.
///
/// Works identically for `cargo run`'s raw binary and the packaged `.app`'s
/// binary — `current_exe()` resolves to the actual executable path in both
/// cases, no bundle-path logic needed. Only exits if the spawn actually
/// succeeded, so a failed relaunch doesn't strand the user with no window at
/// all.
pub(crate) fn relaunch() {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            log::error!("readiness: could not resolve current_exe for relaunch: {e}");
            return;
        }
    };
    match std::process::Command::new(&exe).spawn() {
        Ok(_) => std::process::exit(0),
        Err(e) => log::error!("readiness: failed to relaunch {}: {e}", exe.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_all_covers_every_variant() {
        assert_eq!(Permission::ALL.len(), 3);
        assert!(Permission::ALL.contains(&Permission::Accessibility));
        assert!(Permission::ALL.contains(&Permission::InputMonitoring));
        assert!(Permission::ALL.contains(&Permission::Microphone));
    }

    #[test]
    fn permission_labels_are_distinct_and_nonempty() {
        let labels: Vec<&str> = Permission::ALL.iter().map(|p| p.label()).collect();
        for label in &labels {
            assert!(!label.is_empty());
        }
        let mut unique = labels.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            labels.len(),
            "labels must be distinct: {labels:?}"
        );
    }

    #[test]
    fn permission_descriptions_are_nonempty() {
        for permission in Permission::ALL {
            assert!(!permission.description().is_empty());
        }
    }

    /// Pure, non-prompting: must never panic even in a headless/CI
    /// environment with no TCC state at all.
    #[test]
    fn missing_permissions_does_not_panic() {
        let _ = missing_permissions();
    }

    // ── Pure Access mappings — headless-safe, no macOS/TCC state involved.

    #[test]
    fn mic_access_covers_every_source_variant() {
        assert_eq!(mic_access(MicAuthStatus::Authorized), Access::Granted);
        assert_eq!(mic_access(MicAuthStatus::NotDetermined), Access::Promptable);
        assert_eq!(mic_access(MicAuthStatus::Denied), Access::Denied);
        assert_eq!(mic_access(MicAuthStatus::Restricted), Access::Denied);
    }

    #[test]
    fn input_monitoring_to_access_covers_every_source_variant() {
        assert_eq!(
            input_monitoring_to_access(InputMonitoringAccess::Granted),
            Access::Granted
        );
        assert_eq!(
            input_monitoring_to_access(InputMonitoringAccess::Unknown),
            Access::Promptable
        );
        assert_eq!(
            input_monitoring_to_access(InputMonitoringAccess::Denied),
            Access::Denied
        );
    }

    #[test]
    fn accessibility_access_covers_every_source_variant() {
        assert_eq!(accessibility_access(true), Access::Granted);
        assert_eq!(accessibility_access(false), Access::Promptable);
    }

    /// Every permission's settings URL is a well-formed
    /// `x-apple.systempreferences:` deep-link, and distinct per permission.
    #[test]
    fn settings_urls_are_distinct_deep_links() {
        let urls: Vec<&str> = Permission::ALL.iter().map(|p| p.settings_url()).collect();
        for url in &urls {
            assert!(
                url.starts_with("x-apple.systempreferences:"),
                "not a deep-link: {url}"
            );
        }
        let mut unique = urls.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            urls.len(),
            "settings URLs must be distinct: {urls:?}"
        );
    }

    /// Pure, non-prompting: `Permission::access()` must never panic headless.
    #[test]
    fn permission_access_does_not_panic() {
        for permission in Permission::ALL {
            let _ = permission.access();
        }
    }

    #[test]
    fn format_mb_rounds_to_the_nearest_megabyte() {
        assert_eq!(format_mb(474_000_000), "474 MB");
        assert_eq!(format_mb(496_210_831), "496 MB");
        assert_eq!(format_mb(0), "0 MB");
    }
}
