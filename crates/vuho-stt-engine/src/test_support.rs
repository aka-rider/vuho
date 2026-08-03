//! Shared WAV-fixture test helpers — the ONE place `.wav` parsing lives for
//! this workspace (CONSTITUTION rule 26).
//!
//! Before this module was made `pub` and feature-gated, `test-stt-ffi`
//! (`crates/test-stt-ffi/src/main.rs`) and this crate's `tests/batch_multiwindow.rs`
//! each kept their own byte-identical copy of a RIFF `data`-chunk scanner
//! plus `i16`-PCM-to-`f32` conversion, alongside this module's own
//! `hound`-based loader used by `vad.rs`'s and `stream/session.rs`'s
//! model-gated unit tests — three parsers for one format. All three now
//! call [`load_wav_16k_mono_f32`].
//!
//! Gated behind the `test-fixtures` Cargo feature (see this crate's
//! `Cargo.toml`) so `hound` and this module stay out of default/release
//! builds; `cargo test` on this crate enables the feature via a
//! self-referencing `dev-dependency` (also in `Cargo.toml`), and
//! `test-stt-ffi` / `tests/batch_multiwindow.rs` enable it explicitly on
//! their `vuho-stt-engine` dependency.

use std::path::{Path, PathBuf};

/// Locate `jfk.wav`: `JFK_WAV` env override, else the workspace root.
///
/// The single chokepoint all of this workspace's `jfk.wav`-driven
/// tests/gates resolve the fixture path through — `test-stt-ffi`,
/// `tests/batch_multiwindow.rs`, and this crate's own model-gated unit
/// tests all call this rather than keeping their own copy of the
/// resolution order.
#[must_use]
pub fn jfk_wav_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("JFK_WAV") {
        return Some(PathBuf::from(p));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?.parent()?;
    let candidate = root.join("jfk.wav");
    candidate.is_file().then_some(candidate)
}

/// Errors from [`load_wav_16k_mono_f32`] — a typed replacement for the
/// stringly-typed `Result<_, String>` this used to return (CONSTITUTION
/// rule 31: error conditions should be as precisely typed as the rest of
/// the API), distinguishing "couldn't even open/parse the container" from
/// "a sample inside it failed to decode".
#[derive(thiserror::Error, Debug)]
pub enum WavLoadError {
    /// The file could not be opened or parsed as a WAV container.
    #[error("failed to open WAV file: {0}")]
    Open(#[source] hound::Error),
    /// A sample inside the WAV data could not be decoded.
    #[error("failed to decode WAV sample: {0}")]
    Sample(#[source] hound::Error),
}

/// Load a 16-bit PCM WAV file as 16 kHz mono `f32` samples in `[-1.0, 1.0]`.
///
/// # Errors
///
/// Returns [`WavLoadError`] if the file can't be opened or isn't valid
/// 16-bit PCM WAV — see `hound::WavReader::open`/`samples`.
pub fn load_wav_16k_mono_f32(path: &Path) -> Result<Vec<f32>, WavLoadError> {
    let mut reader = hound::WavReader::open(path).map_err(WavLoadError::Open)?;
    reader
        .samples::<i16>()
        .map(|s| {
            s.map(|v| f32::from(v) / f32::from(i16::MAX))
                .map_err(WavLoadError::Sample)
        })
        .collect()
}

/// Load `jfk.wav` as 16 kHz mono `f32` samples in `[-1.0, 1.0]`, or `None`
/// if the fixture isn't found or fails to parse in this environment
/// (callers should skip with an `eprintln`, not fail).
#[must_use]
pub fn load_jfk_wav_f32() -> Option<Vec<f32>> {
    load_wav_16k_mono_f32(&jfk_wav_path()?).ok()
}
