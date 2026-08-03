//! Parakeet-TDT STT engine via native `CoreML` on the ANE.
//!
//! Loads the Parakeet-TDT model components (Preprocessor, Encoder, Decoder,
//! `RNNTJoint`) as `CoreML` bundles, runs greedy TDT decoding, and exposes
//! a `TranscriptionEngine` trait with batch and streaming paths.
//!
//! VAD uses the embedded Silero v5 from `voice_activity_detector`
//! (crate cannot load external weights — the fetched `models/silero-vad/`
//! exists for a future direct-`ort` v6 swap).

use std::path::{Path, PathBuf};

use vuho_domain::TranscriptionResult;

/// Re-exported so callers that already depend on `vuho-stt-engine` (but not
/// directly on `vuho-audio`) can match on the full microphone TCC status
/// (CONSTITUTION rule 26 — one source of truth for the enum's definition).
pub use vuho_audio::MicAuthStatus;

pub mod vad;

// CoreML inference layer (macOS only).
#[cfg(target_os = "macos")]
mod coreml;

// Parakeet-TDT model components.
mod parakeet;

// Sliding window + merge pipeline.
mod stream;

// Engine handle.
mod engine;

// Shared WAV-fixture test helpers (jfk.wav loading + generic WAV parsing) —
// the one place this workspace's WAV parsing lives (CONSTITUTION rule 26),
// reused by this crate's own model-gated unit tests (`vad.rs`,
// `stream/session.rs`), `tests/batch_multiwindow.rs`, and `test-stt-ffi`.
// Gated behind the `test-fixtures` feature so it (and its `hound`
// dependency) stay out of default/release builds — see this module's doc
// comment.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_support;

// Re-exports of pure, model-free internals for `benches/hot_paths.rs`. A
// criterion bench target is compiled as a separate crate, so it only sees
// this crate's `pub` surface — `tdt_greedy`, `StepModel`, `TokenAt`,
// `DecoderState`, `merge`, `MergeOutcome`, `windower::plan`, and
// `windower::OVERLAP_FRAMES` are marked `pub` (not `pub(crate)`) at their
// own declaration sites specifically so this module can re-export them (a
// `pub use` cannot widen a `pub(crate)` item's visibility — only forward an
// already-`pub` one), with a doc comment there pointing back here. Their
// FIELDS stay `pub(crate)` — the bench never needs to read/construct them
// directly, only via the shim constructors below, which live inside this
// crate and so can see those fields.
pub mod bench_support {
    //! Re-exports + constructor shims for `benches/hot_paths.rs`. Nothing
    //! here touches `CoreML` — every benched function is model-free (a
    //! synthetic `StepModel`/token stream stands in for the real engine),
    //! matching the plan's "model-free, deterministic" bench requirement.
    pub use crate::parakeet::decoder_state::DecoderState;
    pub use crate::parakeet::tdt::{tdt_greedy, StepModel, TokenAt};
    pub use crate::stream::merge::{merge, MergeOutcome};
    pub use crate::stream::windower::{plan, OVERLAP_FRAMES};
    pub use crate::EngineError;

    /// Construct a `TokenAt` — the bench's only way to build one (see this
    /// module's doc comment).
    #[must_use]
    pub fn token_at(id: u32, frame: usize) -> TokenAt {
        TokenAt { id, frame }
    }

    /// A `StepModel` that always emits a fixed `token` at a fixed
    /// `duration`, regardless of frame or decoder state — the same fixture
    /// `parakeet::tdt`'s own unit tests use (`FixedStepModel`), exposed here
    /// so `benches/hot_paths.rs` can drive `tdt_greedy` without a real
    /// model.
    struct FixedStepModel {
        token: u32,
        duration: u32,
    }

    impl StepModel for FixedStepModel {
        fn decode(&self, _token: i32, state: &mut DecoderState) -> Result<(), EngineError> {
            state.dec_out = Some(vec![0.0; 640]);
            Ok(())
        }

        fn joint(
            &self,
            _enc_frame: &[f32],
            _dec_out: &[f32],
            out: &mut Vec<f32>,
        ) -> Result<(), EngineError> {
            out.clear();
            out.resize(8198, f32::NEG_INFINITY);
            out[self.token as usize] = 0.0;
            for (i, bin) in out[8193..8198].iter_mut().enumerate() {
                #[allow(clippy::cast_possible_truncation)] // i is always 0..5
                let i = i as u32;
                *bin = if i == self.duration {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
            }
            Ok(())
        }
    }

    /// Build a `StepModel` fixture for the bench (see `FixedStepModel`, private).
    ///
    /// Returns `impl StepModel` (a concrete, `Sized` type), not
    /// `Box<dyn StepModel>` — `tdt_greedy` takes `&impl StepModel`, which
    /// requires `Sized`; a trait object reference doesn't satisfy that.
    #[must_use]
    pub fn fixed_step_model(token: u32, duration: u32) -> impl StepModel {
        FixedStepModel { token, duration }
    }
}

// ── Errors ────────────────────────────────────────────────────────────
/// Errors from loading or running the STT engine.
#[derive(thiserror::Error, Debug, Clone)]
pub enum EngineError {
    /// No model folder was found at any of the resolution chain's candidate
    /// locations.
    #[error(
        "model folder not found — tried: {} — provision the model at one of these locations, or point VUHO_MODEL_FOLDER at an existing model directory",
        tried.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    )]
    ModelFolderMissing {
        /// Every candidate path the resolver tried, in the order it tried
        /// them (env override → bundle → workspace dev → user-data — see
        /// [`vuho_model_paths::resolve_model_folder`]).
        tried: Vec<PathBuf>,
    },
    /// Loading the `CoreML` model components failed.
    #[error("model load failed: {0}")]
    LoadFailed(String),
    /// A `CoreML`-level failure: prediction, feature-provider construction,
    /// output-feature extraction, or `MLMultiArray` allocation/shape/dtype
    /// handling. Distinct from [`Self::Transcribe`], which is reserved for
    /// failures in this crate's own transcription-algorithm code (the
    /// sliding-window/merge pipeline), not in a `CoreML` call itself.
    #[error("CoreML error: {0}")]
    CoreMl(String),
    /// A batch or streaming transcription call failed for a reason that
    /// isn't a `CoreML` call itself (see [`Self::CoreMl`]) — e.g. output
    /// shape/length invariants the decode algorithm expects but a model
    /// didn't satisfy.
    #[error("transcription failed: {0}")]
    Transcribe(String),
    /// The microphone permission (TCC) was denied.
    #[error("microphone permission denied")]
    MicPermissionDenied,
    /// `stop_stream` was called with no streaming session active.
    #[error("no active streaming session")]
    NoActiveStream,
    /// `start_stream` was called while a streaming session was already
    /// active. Previously this was documented as "undefined behavior" on
    /// the [`TranscriptionEngine`] trait; the real implementation
    /// ([`ParakeetEngine`]) has always returned this typed error on
    /// double-start, so the doc now matches the code.
    #[error("a streaming session is already active")]
    StreamAlreadyActive,
    /// The streaming session's background thread panicked instead of
    /// returning normally.
    #[error("streaming session thread panicked")]
    SessionPanicked,
    /// The OS failed to spawn the streaming session's background thread.
    #[error("failed to spawn streaming session thread: {0}")]
    SpawnFailed(String),
    /// Starting audio capture for a streaming session failed (any
    /// [`vuho_audio::AudioError`] other than `PermissionDenied`, which maps
    /// to [`Self::MicPermissionDenied`] instead).
    #[error("audio capture error: {0}")]
    Audio(#[from] vuho_audio::AudioError),
}

// ── Model folder resolution ───────────────────────────────────────────
//
// The env-var → bundle → workspace fallback chain itself lives in
// `vuho-model-paths` (a chokepoint — `vuho_model_paths::resolve_model_folder`
// — shared by every model this workspace loads); this crate only supplies
// the STT entry of the embedded `models.manifest.json` and wraps the result
// in its own `EngineError`.

/// Resolve the model directory — the single place any caller learns where the
/// model lives.
///
/// Order: `VUHO_MODEL_FOLDER` → the enclosing `.app`'s `Contents/Resources/`
/// → the workspace-relative dev directory → `~/Library/Application
/// Support/Vuho/models/<name>` (see
/// [`vuho_model_paths::resolve_model_folder`] for the full rationale). This
/// crate's own resolution never downloads — a model that isn't on disk at
/// any candidate is an error here rather than a silent network fetch; only
/// `vuho-model-fetch`, gated behind an explicit user action, ever performs
/// network I/O (ADR-020).
///
/// # Errors
///
/// Returns `EngineError::ModelFolderMissing` naming every candidate tried
/// if none of them is an existing directory.
pub fn resolve_model_folder() -> Result<PathBuf, EngineError> {
    vuho_model_paths::resolve_model_folder(&vuho_model_paths::manifest().stt.spec())
        .map_err(|e| EngineError::ModelFolderMissing { tried: e.tried })
}

/// Required sub-paths inside a model directory for the parakeet TDT model —
/// the STT component list from the embedded `models.manifest.json`, the one
/// place this list is written down (`scripts/*.sh` read the same file).
///
/// The `FluidInference/parakeet-tdt-0.6b-v3-coreml` model ships these `CoreML`
/// component bundles plus a vocabulary file. If any are missing the engine
/// cannot transcribe.
fn required_components() -> &'static [String] {
    &vuho_model_paths::manifest().stt.components
}

/// Validate that a model directory contains all required Parakeet-TDT `CoreML` components.
///
/// Returns `Ok(())` when every component exists. On failure, returns
/// `EngineError::LoadFailed` naming the **first** missing component so the
/// user can fix the model layout in one pass.
///
/// # Errors
///
/// `EngineError::LoadFailed` with a message like
/// `"missing model component: TextDecoderContextPrefill.mlmodelc"` when
/// at least one required component is absent.
pub fn validate_model_layout(model_dir: &Path) -> Result<(), EngineError> {
    for component in required_components() {
        if !model_dir.join(component).exists() {
            return Err(EngineError::LoadFailed(format!(
                "missing model component: {component}"
            )));
        }
    }
    Ok(())
}

// ── TranscriptionEngine trait (the STT port from ADR-004) ─────────────
/// A transcription engine that is **already initialized with its models
/// loaded** — construction is the only place that cost is paid.
///
/// There is deliberately no `init`/`load_models` here: an implementor hands
/// out instances only once they are ready (see [`ParakeetEngine::load`]), so
/// no caller can hold an engine in a half-built state or trigger a multi-minute
/// model load from a hot path.
pub trait TranscriptionEngine {
    /// Transcribe a buffer of 16 kHz mono f32 samples (blocking).
    ///
    /// # Errors
    ///
    /// Returns `EngineError::CoreMl` if a `CoreML` call fails, or
    /// `EngineError::Transcribe` if the decode algorithm's own invariants
    /// (e.g. an output length it requires) are violated.
    fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, EngineError>;

    /// Unload models and release resources.
    fn unload(&self);

    /// Start a streaming transcription session. Returns a channel receiver for
    /// live `DictationEvent`s.
    ///
    /// `input_device` is the configured microphone device **name**; `None`
    /// uses the system default input, as does an unresolvable (e.g.
    /// unplugged) name.
    ///
    /// Single-stream-only: calling while a stream is already active returns
    /// `Err(EngineError::StreamAlreadyActive)` rather than replacing (and
    /// thereby leaking) the live session — the session started by the
    /// first, still-active call is left running untouched.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::MicPermissionDenied` if microphone access is
    /// denied. Returns `EngineError::StreamAlreadyActive` if a stream is
    /// already active. Returns `EngineError::Audio` if capture fails to
    /// start, or `EngineError::SpawnFailed` if the session thread itself
    /// cannot be spawned.
    fn start_stream(
        &self,
        language: Option<&str>,
        input_device: Option<&str>,
    ) -> Result<crossbeam_channel::Receiver<vuho_domain::DictationEvent>, EngineError>;

    /// Stop the active streaming session and return the final transcription.
    ///
    /// After this call, the `Receiver` returned by `start_stream` is closed
    /// — drain any remaining events.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::NoActiveStream` if no stream was started.
    /// Returns `EngineError::SessionPanicked` if the streaming session's
    /// background thread panicked instead of finishing normally.
    fn stop_stream(&self) -> Result<TranscriptionResult, EngineError>;
}

// ── ParakeetEngine (the real engine) ───────────────────────────────────

pub use engine::ParakeetEngine;

/// List the names of available audio input devices.
///
/// # Errors
///
/// Returns `EngineError::Audio` if the host cannot enumerate input devices
/// at all (not if the list is merely empty).
pub fn list_input_devices() -> Result<Vec<String>, EngineError> {
    Ok(vuho_audio::list_input_device_names()?)
}

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
    use vuho_audio::MicAuthStatus;
    match vuho_audio::mic_authorization_status() {
        MicAuthStatus::Authorized => true,
        MicAuthStatus::NotDetermined => {
            vuho_audio::request_mic_access_async();
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
    vuho_audio::mic_authorization_status()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// `load` is the only constructor, so a bad model folder yields no engine
    /// at all rather than one that fails later, mid-session.
    #[test]
    fn engine_load_fails_for_a_missing_model_folder() {
        assert!(ParakeetEngine::load(PathBuf::from("/nonexistent-model")).is_err());
    }

    #[test]
    fn resolve_model_folder_honors_env_override() {
        // `VUHO_MODEL_FOLDER` is terminal, not merely tried first
        // (ADR-020/ADR-019): a nonexistent override must be the *only*
        // candidate tried, never falling through to the bundle/dev/user
        // candidates — see `vuho_model_paths`'s
        // `missing_env_override_errors_instead_of_falling_through_to_dev_dir`,
        // which this test mirrors. Asserting only `tried.first()` would
        // pass under a fall-through implementation too and not falsify the
        // regression this test exists to guard.
        // SAFETY: single-threaded test; no other thread reads the environment.
        unsafe {
            std::env::set_var("VUHO_MODEL_FOLDER", "/definitely/not/a/model");
            std::env::set_var(
                "VUHO_MODEL_NAME",
                "vuho-stt-engine-test-name-that-cannot-exist",
            );
        }
        let err = resolve_model_folder().expect_err("nonexistent folder must not resolve");
        unsafe {
            std::env::remove_var("VUHO_MODEL_FOLDER");
            std::env::remove_var("VUHO_MODEL_NAME");
        }

        match err {
            EngineError::ModelFolderMissing { tried } => {
                assert_eq!(
                    tried,
                    vec![PathBuf::from("/definitely/not/a/model")],
                    "the override must be the only candidate tried"
                );
            }
            other => panic!("expected ModelFolderMissing, got {other:?}"),
        }
    }

    /// A directory with all required components passes validation.
    #[test]
    fn validate_model_layout_passes_when_all_components_present() {
        let tmp = std::env::temp_dir().join("vuho-test-model-ok");
        fs::remove_dir_all(&tmp).ok();
        fs::create_dir_all(&tmp).expect("create tempdir");
        for comp in required_components() {
            let p = tmp.join(comp);
            if comp.ends_with(".mlmodelc") {
                fs::create_dir_all(&p).expect("create component dir");
            } else {
                fs::write(&p, "").expect("create component file");
            }
        }
        assert!(validate_model_layout(&tmp).is_ok());
        fs::remove_dir_all(&tmp).ok();
    }

    /// Validation fails on the first missing component, and the error names it.
    #[test]
    fn validate_model_layout_fails_on_first_missing_component() {
        let tmp = std::env::temp_dir().join("vuho-test-model-missing");
        fs::remove_dir_all(&tmp).ok();
        fs::create_dir_all(&tmp).expect("create tempdir");
        // Only create the first two components — ParakeetEncoder_15s is missing.
        fs::create_dir_all(tmp.join("Preprocessor.mlmodelc")).unwrap();
        fs::create_dir_all(tmp.join("ParakeetEncoder_15s.mlmodelc")).unwrap();

        let err = validate_model_layout(&tmp).expect_err("should fail");
        match err {
            EngineError::LoadFailed(msg) => {
                assert!(
                    msg.contains("ParakeetDecoder.mlmodelc"),
                    "error should name ParakeetDecoder.mlmodelc, got: {msg}"
                );
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
        fs::remove_dir_all(&tmp).ok();
    }

    /// An empty directory fails on the very first required component.
    #[test]
    fn validate_model_layout_fails_on_empty_dir() {
        let tmp = std::env::temp_dir().join("vuho-test-model-empty");
        fs::remove_dir_all(&tmp).ok();
        fs::create_dir_all(&tmp).expect("create tempdir");

        let err = validate_model_layout(&tmp).expect_err("should fail");
        match err {
            EngineError::LoadFailed(msg) => {
                assert!(
                    msg.contains("Preprocessor.mlmodelc"),
                    "should name first component, got: {msg}"
                );
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
        fs::remove_dir_all(&tmp).ok();
    }
}
