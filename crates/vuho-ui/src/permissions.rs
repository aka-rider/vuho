//! macOS permission prompts (ADR-012 / GAP 3; preflight gate: ADR-016).
//!
//! Accessibility (for the `CapsLock` `CGEventTap`) and Microphone (for STT
//! capture) are runtime TCC grants, not entitlements. As of ADR-016, the
//! primary path for both is the startup preflight gate
//! (`permission_gate::open_gate_window`) — the functions here are now only
//! defensive fallbacks for the two reactive call sites where a grant is
//! revoked **mid-session** (`start_hotkey`'s error branch,
//! `settings_window.rs`'s `select_hotkey` error branch), so a single native
//! dialog (or none, if still trusted) is the correct, non-nagging behavior
//! there too — never native+custom dialog stacking.
//!
//! All functions must run on the main thread (they build `AppKit` UI).

use objc2::MainThreadMarker;
use objc2_app_kit::{NSAlert, NSApplication, NSWorkspace};
use objc2_foundation::{NSString, NSURL};

/// `NSModalResponse` for the first (default) alert button.
const NS_ALERT_FIRST_BUTTON: isize = 1000;

/// System Settings deep-link anchors (ADR-016) — the one place every
/// permission's settings-pane URL lives (CONSTITUTION rule 26). `pub(crate)`
/// so `permission_gate.rs`'s denied-state "Open System Settings" buttons
/// reuse these instead of duplicating URL strings.
pub(crate) const MICROPHONE_SETTINGS_URL: &str =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone";

/// Confirmed via `AXIsProcessTrustedWithOptions`'s own dialog and widely
/// documented `x-apple.systempreferences` anchor conventions. Only consumed
/// by `permission_gate.rs` — cfg-gated so it isn't dead code under
/// `--features demo`, which never builds that module.
#[cfg(not(feature = "demo"))]
pub(crate) const ACCESSIBILITY_SETTINGS_URL: &str =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility";

/// `Privacy_ListenEvent` is the TCC service key for Input Monitoring, found
/// in `/System/Library/PreferencePanes/Security.prefPane/Contents/Resources/PrivacyTCCServices.plist`
/// (the "Input Monitoring" pane has no scriptable UI element to derive the
/// anchor from directly, unlike Microphone/Accessibility — this anchor was
/// verified against public macOS deep-link references, not driven
/// end-to-end on this development machine; re-confirm by clicking through
/// once a permission is actually in the Denied state). Only consumed by
/// `permission_gate.rs` — cfg-gated for the same reason as
/// `ACCESSIBILITY_SETTINGS_URL` above.
#[cfg(not(feature = "demo"))]
pub(crate) const INPUT_MONITORING_SETTINGS_URL: &str =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent";

/// Prompt for Accessibility permission (needed for the `CapsLock` event tap).
///
/// Triggers only the native system dialog (which also registers the app in
/// the Accessibility list) — no additional custom alert. `AXIsProcessTrustedWithOptions`
/// returns immediately without waiting for the user's answer, and only shows
/// a dialog at all if not already trusted, so this never stacks a second
/// dialog on top of it.
///
/// Only called from `wire_production` (production-only wiring) — cfg-gated so
/// it isn't dead code under `--features demo`, which has no hotkey to prompt for.
#[cfg(not(feature = "demo"))]
pub(crate) fn prompt_accessibility() {
    let _ = vuho_os_integration::prompt_accessibility_trust();
}

/// Alert the user that microphone access is required, with a settings shortcut.
///
/// Called reactively when the pipeline reports a microphone-permission error
/// (see the typed `ErrorKind::MicPermissionDenied` path).
///
/// # Contract: must be called from a deferred/async context, never
/// synchronously inside the top-level `Application::run` closure
///
/// This function's `NSAlert::runModal()` pumps a nested run loop, exactly
/// like `prompt_accessibility`'s underlying dialog — see
/// `wiring::start_hotkey`'s doc comment for the full hazard: while
/// `Application::run`'s closure is still on the stack, the app context is
/// borrowed for its entire duration, and anything that re-enters GPUI from
/// within a nested modal run loop during that window (e.g. the overlay's
/// own animation timer) hits an already-borrowed panic. Its current caller,
/// `event_loop::apply_events`, is safe: it only ever runs from inside
/// `spawn_event_drain`'s `cx.spawn`'d task, which GPUI's
/// `ForegroundExecutor` dispatches to run *after* the top-level closure has
/// already returned — never call this function any earlier in that
/// closure's own call stack.
pub(crate) fn show_microphone_denied() {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    activate_app(mtm);

    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str("Vuho needs Microphone access"));
    alert.setInformativeText(&NSString::from_str(
        "Enable Vuho under System Settings → Privacy & Security → Microphone to dictate.",
    ));
    alert.addButtonWithTitle(&NSString::from_str("Open Settings"));
    alert.addButtonWithTitle(&NSString::from_str("Later"));

    if alert.runModal() == NS_ALERT_FIRST_BUTTON {
        open_url(MICROPHONE_SETTINGS_URL);
    }
}

/// Bring the accessory app forward so a modal alert (or a focused window,
/// such as the permission gate) appears frontmost.
///
/// R2: `activate` is inherited (not overridden) on GPUI's `NSApplication`
/// subclass, so the typed call is safe here — unlike `setActivationPolicy:` in
/// `window_config.rs`, which GPUI overrides with a mismatched return-type
/// encoding and thus requires raw `objc_msgSend`. (`activateIgnoringOtherApps:`
/// is deprecated; `activate` is the modern equivalent.)
///
/// `pub(crate)`: also the chokepoint `permission_gate` uses to front its
/// window on launch (and again on every "Permissions…" menu click) — the app
/// runs under `NSApplicationActivationPolicyAccessory`, so a focused window
/// alone does not bring the app forward without this.
pub(crate) fn activate_app(mtm: MainThreadMarker) {
    let app = NSApplication::sharedApplication(mtm);
    app.activate();
}

/// Open a URL via `NSWorkspace` (used for the System Settings deep-links).
///
/// `pub(crate)` so `permission_gate.rs` reuses this opener rather than
/// writing a second one (CONSTITUTION rule 26).
pub(crate) fn open_url(url: &str) {
    if let Some(ns_url) = NSURL::URLWithString(&NSString::from_str(url)) {
        let _ = NSWorkspace::sharedWorkspace().openURL(&ns_url);
    }
}
