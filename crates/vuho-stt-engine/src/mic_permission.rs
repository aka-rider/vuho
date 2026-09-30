//! macOS microphone permission (TCC). Other platforms have no such concept
//! — capture either works or fails inside `vuho_audio::start_capture` — so
//! this whole module exists on macOS only.

use vuho_audio::{mic_authorization_status, request_mic_access_async};

use crate::EngineError;

/// Re-exported so callers that already depend on `vuho-stt-engine` (but not
/// directly on `vuho-audio`) can match on the full microphone TCC status
/// (CONSTITUTION rule 26 — one source of truth for the enum's definition).
pub use vuho_audio::MicAuthStatus;

/// Check the current microphone permission status, prompting if it has
/// never been asked.
///
/// If the status is not yet determined, this also triggers the system TCC
/// dialog (`request_mic_access_async`) so the caller doesn't need a second,
/// separate call to prompt — but the dialog is asynchronous and this
/// function does not wait for the user's answer, so a `NotDetermined`
/// result here always returns `false` even if the user is about to grant
/// access; re-check on the next session start.
///
/// This is infallible (no TCC query used here can fail in a way this crate
/// distinguishes) — collapsed from a vestigial `Result<bool, EngineError>`
/// that no caller ever matched an `Err` arm on.
///
/// # Returns
///
/// `true` if the user has already granted microphone access, `false` if
/// denied, restricted, or not yet determined.
#[must_use]
pub fn request_mic_permission() -> bool {
    match mic_authorization_status() {
        MicAuthStatus::Authorized => true,
        MicAuthStatus::NotDetermined => {
            request_mic_access_async();
            false
        }
        MicAuthStatus::Denied | MicAuthStatus::Restricted => false,
    }
}

/// Pure (non-prompting) microphone permission status.
///
/// Unlike [`request_mic_permission`], this never triggers the native TCC
/// dialog even when the status is `NotDetermined` — used by the startup
/// preflight permission gate (ADR-016), which must be side-effect-free on
/// its initial check, matching the Accessibility/Input Monitoring checks it
/// runs alongside. The gate distinguishes `NotDetermined` (promptable) from
/// `Denied`/`Restricted` (only fixable via System Settings), which a
/// collapsed bool cannot express — which is why this crate exposes exactly
/// two mic accessors (this one and [`request_mic_permission`]), not three:
/// a third bool-only projection of this same status used to exist and had
/// zero callers.
#[must_use]
pub fn mic_permission_status() -> MicAuthStatus {
    mic_authorization_status()
}

/// Synchronous precheck: a known-denied/restricted status fails
/// immediately, without spawning a capture thread that would only fail
/// moments later. `NotDetermined` proceeds — macOS raises the TCC dialog
/// itself on the first real capture attempt inside
/// `vuho_audio::start_capture` (see `vuho-ui`'s
/// `request_mic_permission_on_startup` doc comment).
pub(crate) fn ensure_not_denied() -> Result<(), EngineError> {
    match mic_authorization_status() {
        MicAuthStatus::Denied | MicAuthStatus::Restricted => Err(EngineError::MicPermissionDenied),
        MicAuthStatus::Authorized | MicAuthStatus::NotDetermined => Ok(()),
    }
}
