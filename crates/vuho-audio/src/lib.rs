//! Rust-native microphone capture: cpal capture thread + resample to 16 kHz
//! mono. No `vuho-*` dependencies — this crate is a leaf the STT engine
//! consumes, not the other way around.
//!
//! Capture is owned entirely by a dedicated thread (`"vuho-audio-capture"`)
//! because `cpal::Stream` is `!Send`; it must be built, played, and dropped
//! on the same thread (CONSTITUTION rule 1: one owner per resource).

mod capture;
#[cfg(target_os = "macos")]
mod permission;
mod resample;

pub use capture::{start_capture, CaptureHandle};
#[cfg(target_os = "macos")]
pub use permission::{mic_authorization_status, request_mic_access_async, MicAuthStatus};

/// Output sample rate every capture stream resamples to.
pub const OUTPUT_SAMPLE_RATE: u32 = 16_000;

/// Errors from audio capture / device enumeration.
#[derive(thiserror::Error, Debug, Clone)]
pub enum AudioError {
    /// The user (or a prior TCC decision) denied microphone access.
    #[error("microphone permission denied")]
    PermissionDenied,
    /// The configured (or default) input device could not be resolved.
    #[error("audio device unavailable: {0}")]
    DeviceUnavailable(String),
    /// `cpal` failed to build the input stream for the device/config.
    #[error("failed to build audio stream: {0}")]
    StreamBuild(String),
    /// `cpal` failed to start (`.play()`) the built input stream.
    #[error("failed to start audio stream: {0}")]
    StreamPlay(String),
    /// The stream's error callback fired — capture stopped unexpectedly.
    #[error("audio stream died: {0}")]
    StreamDied(String),
    /// `rubato` resampling to [`OUTPUT_SAMPLE_RATE`] failed.
    #[error("resampling failed: {0}")]
    Resample(String),
}

/// Capture configuration. Output is always mono @ [`OUTPUT_SAMPLE_RATE`] Hz.
#[derive(Debug, Clone, Default)]
pub struct CaptureConfig {
    /// Device name to capture from (as reported by [`list_input_device_names`]).
    /// `None`, or a name that no longer resolves (e.g. unplugged), falls back
    /// to the system default input device.
    pub device_name: Option<String>,
}

/// List the names of available audio input devices.
///
/// # Errors
///
/// Returns [`AudioError::DeviceUnavailable`] if the host cannot enumerate
/// input devices at all (not if the list is merely empty).
pub fn list_input_device_names() -> Result<Vec<String>, AudioError> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .map_err(|e| AudioError::DeviceUnavailable(e.to_string()))?;
    Ok(devices
        .filter_map(|d| d.description().ok().map(|desc| desc.name().to_owned()))
        .collect())
}
