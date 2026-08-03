//! macOS microphone (TCC) permission status, via `AVCaptureDevice`.
//!
//! macOS also raises the system permission dialog automatically on first
//! capture attempt (see `crates/vuho-stt-engine` mic-permission-handling
//! notes) — this module exists so the settings UI and pipeline can *ask*
//! the state up front rather than only discovering denial via a failed
//! stream build.

use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeAudio};

/// Mirrors `AVAuthorizationStatus` for the audio media type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicAuthStatus {
    /// The user has granted microphone access.
    Authorized,
    /// The user has not yet been asked (no TCC decision recorded).
    NotDetermined,
    /// The user explicitly denied microphone access.
    Denied,
    /// Access is restricted by system policy (e.g. parental controls, MDM)
    /// and cannot be changed by the user via a prompt.
    Restricted,
}

/// Current microphone authorization status. Does not prompt.
///
/// # Panics
///
/// Panics if the `AVMediaTypeAudio` constant isn't linked (would indicate a
/// broken `AVFoundation` framework load, not a runtime condition).
#[must_use]
pub fn mic_authorization_status() -> MicAuthStatus {
    // SAFETY: `AVMediaTypeAudio` is a valid static `AVMediaType` constant;
    // this class method reads TCC state and does not prompt.
    let media_type = unsafe { AVMediaTypeAudio }.expect("AVMediaTypeAudio constant must be linked");
    let status = unsafe { AVCaptureDevice::authorizationStatusForMediaType(media_type) };
    match status {
        AVAuthorizationStatus::Authorized => MicAuthStatus::Authorized,
        AVAuthorizationStatus::Denied => MicAuthStatus::Denied,
        AVAuthorizationStatus::Restricted => MicAuthStatus::Restricted,
        _ => MicAuthStatus::NotDetermined,
    }
}

/// Trigger the system permission dialog if status is `NotDetermined`.
/// Fire-and-forget: the completion handler is a no-op because the caller
/// re-checks [`mic_authorization_status`] on the next session start rather
/// than awaiting this call (the dialog is modal and blocks on user input,
/// which can be arbitrarily long).
///
/// # Panics
///
/// Panics if the `AVMediaTypeAudio` constant isn't linked (see
/// [`mic_authorization_status`]).
pub fn request_mic_access_async() {
    let media_type = unsafe { AVMediaTypeAudio }.expect("AVMediaTypeAudio constant must be linked");
    // SAFETY: the completion block captures no state and is valid for the
    // duration of the (asynchronous) Objective-C call; AVFoundation invokes
    // it exactly once on an arbitrary queue.
    let handler = block2::RcBlock::new(move |_granted: objc2::runtime::Bool| {});
    unsafe {
        AVCaptureDevice::requestAccessForMediaType_completionHandler(media_type, &handler);
    }
}
