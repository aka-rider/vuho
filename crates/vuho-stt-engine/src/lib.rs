//! STT engines via native `CoreML`, behind one `TranscriptionEngine` trait
//! with batch and streaming paths.
//!
//! Two backends share everything but the decode loop, meeting at the
//! `WindowInference` seam (ADR-022) so `StreamingEngine<M>` owns the window
//! planning, the seam merge, and the session lifecycle exactly once:
//!
//! - **Parakeet-TDT** (`ParakeetEngine`, the manifest's default model):
//!   Preprocessor, Encoder, Decoder, and `RNNTJoint` `CoreML` bundles,
//!   greedy TDT decoding over a sliding 15 s window. Its encoder runs on
//!   the ANE.
//! - **Canary-1B-v2** (`CanaryEngine`): Preprocessor, Encoder, Decoder, and
//!   Projection bundles, greedy attention encoder-decoder over a fixed 15 s
//!   window. Every component loads CPU-only — measured, not assumed; see
//!   `canary::models::Components::load`.
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

// Canary-1B-v2 model components. `pub` for its `prompt` module alone: the
// Settings UI needs the language list this backend can actually reach, and
// a second copy of that list anywhere else would be a list to keep in sync
// (CONSTITUTION rule 26).
pub mod canary;

// The backend-independent engine half: batch windowing + session lifecycle.
mod streaming_engine;

// The backend-independent decoded-token type and frame->ms conversion.
pub(crate) mod token;

// Vocabulary loading + detokenization, shared by every backend.
pub(crate) mod vocab;

// The seam every STT backend implements for the shared streaming pipeline.
pub(crate) mod window_inference;

// Sliding window + merge pipeline.
mod stream;

// Engine handles, one thin wrapper per backend.
mod canary_engine;
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
// `frame_ms`, `DecoderState`, `merge`, `MergeBounds`, `MergeOutcome`,
// `windower::plan`, and `windower::OVERLAP_FRAMES` are marked `pub` (not
// `pub(crate)`) at their own declaration sites specifically so this module can re-export them (a
// `pub use` cannot widen a `pub(crate)` item's visibility — only forward an
// already-`pub` one), with a doc comment there pointing back here. Their
// FIELDS stay `pub(crate)` — the bench never needs to read/construct them
// directly, only via the shim constructors below, which live inside this
// crate and so can see those fields.
pub mod bench_support {
    //! Re-exports + constructor shims for this crate's separate bench and
    //! integration-test crates (`benches/hot_paths.rs`,
    //! `tests/canary_batch.rs`) — the one sanctioned way they reach a
    //! crate-internal item, instead of restating its value as a literal
    //! (CONSTITUTION rule 27). Nothing here touches `CoreML` — every benched
    //! function is model-free (a synthetic `StepModel`/token stream stands
    //! in for the real engine), matching the plan's "model-free,
    //! deterministic" bench requirement.
    pub use crate::parakeet::decoder_state::DecoderState;
    pub use crate::parakeet::tdt::{tdt_greedy, StepModel};
    pub use crate::stream::merge::{merge, MergeBounds, MergeOutcome};
    pub use crate::stream::windower::{plan, OVERLAP_FRAMES, WINDOW_SAMPLES};
    pub use crate::token::{frame_ms, TokenAt};
    pub use crate::EngineError;

    /// Construct a `TokenAt` — the bench's only way to build one (see this
    /// module's doc comment).
    #[must_use]
    pub fn token_at(id: u32, pos: usize) -> TokenAt {
        TokenAt { id, pos }
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
    /// The model directory name — usually a `VUHO_MODEL_NAME` override — is
    /// not a single plain path component, so no candidate location could be
    /// built from it. Distinct from [`Self::ModelFolderMissing`]: nothing
    /// was tried, because the name itself is unusable.
    #[error("{0}")]
    InvalidModelDirName(String),
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
    /// The selected model cannot transcribe the session's language. Carries
    /// both facts so the UI can name them without re-deriving either
    /// (CONSTITUTION rule 2: no silent fallback to some other language).
    #[error("{model} does not support {language}")]
    UnsupportedLanguage {
        /// The selected model's display name.
        model: String,
        /// The BCP-47-derived language code that was asked for.
        language: String,
    },
    /// The requested model id is absent from the embedded
    /// `models.manifest.json`, so there is no directory name, revision, or
    /// asset list to resolve it with.
    #[error("unknown model id: {0}")]
    UnknownModel(String),
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
/// Returns `EngineError::UnknownModel` if `model_id` names no model in the
/// embedded manifest, `EngineError::InvalidModelDirName` if the directory
/// name (or its `VUHO_MODEL_NAME` override) is not a single plain path
/// component, or `EngineError::ModelFolderMissing` naming every candidate
/// tried if none of them is an existing directory.
pub fn resolve_model_folder(model_id: &str) -> Result<PathBuf, EngineError> {
    let spec = vuho_model_paths::manifest()
        .stt
        .spec_for(model_id)
        .ok_or_else(|| EngineError::UnknownModel(model_id.to_owned()))?;
    vuho_model_paths::resolve_model_folder(&spec).map_err(|e| match e {
        vuho_model_paths::ModelPathError::NotFound { tried } => {
            EngineError::ModelFolderMissing { tried }
        }
        vuho_model_paths::ModelPathError::InvalidDirName(invalid) => {
            EngineError::InvalidModelDirName(invalid.to_string())
        }
    })
}

/// Required sub-paths inside `model_id`'s directory — the model's asset
/// filenames from the embedded `models.manifest.json`, the one place this
/// list is written down (`scripts/*.sh` read the same file).
///
/// # Errors
///
/// Returns `EngineError::UnknownModel` if `model_id` names no model in the
/// embedded manifest.
fn required_components(model_id: &str) -> Result<Vec<&'static str>, EngineError> {
    vuho_model_paths::manifest()
        .stt
        .model(model_id)
        .map(vuho_model_paths::SttModel::components)
        .ok_or_else(|| EngineError::UnknownModel(model_id.to_owned()))
}

/// The asset roles a backend loads by name from the embedded manifest.
///
/// The manifest maps each role to a filename; these role keys are the only
/// model-file identifiers written down in Rust (ADR-019 keeps every actual
/// `.mlmodelc`/vocabulary filename in `models.manifest.json`).
pub(crate) mod asset_role {
    pub(crate) const PREPROCESSOR: &str = "preprocessor";
    pub(crate) const ENCODER: &str = "encoder";
    pub(crate) const DECODER: &str = "decoder";
    pub(crate) const JOINT: &str = "joint";
    pub(crate) const PROJECTION: &str = "projection";
    pub(crate) const VOCAB: &str = "vocab";
}

/// Path to `model_id`'s `role` asset inside `model_dir` — the one place a
/// backend turns a role into a file (ADR-019: filenames live in the
/// manifest, never in this crate).
///
/// # Errors
///
/// Returns `EngineError::UnknownModel` for an unknown `model_id`, or
/// `EngineError::LoadFailed` if the model declares no asset for `role`.
pub(crate) fn asset_path(
    model_id: &str,
    role: &str,
    model_dir: &Path,
) -> Result<PathBuf, EngineError> {
    let model = vuho_model_paths::manifest()
        .stt
        .model(model_id)
        .ok_or_else(|| EngineError::UnknownModel(model_id.to_owned()))?;
    let file = model.asset(role).ok_or_else(|| {
        EngineError::LoadFailed(format!("model {model_id} declares no '{role}' asset"))
    })?;
    Ok(model_dir.join(file))
}

/// Validate that `model_dir` contains every asset `model_id` declares.
///
/// Returns `Ok(())` when every component exists. On failure, returns
/// `EngineError::LoadFailed` naming the **first** missing component so the
/// user can fix the model layout in one pass.
///
/// # Errors
///
/// `EngineError::UnknownModel` for an unknown `model_id`, or
/// `EngineError::LoadFailed` with a message like
/// `"missing model component: RNNTJoint.mlmodelc"` when at least one
/// required component is absent.
pub fn validate_model_layout(model_id: &str, model_dir: &Path) -> Result<(), EngineError> {
    for component in required_components(model_id)? {
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

pub use canary_engine::CanaryEngine;
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
        assert!(
            ParakeetEngine::load(default_model_id(), PathBuf::from("/nonexistent-model")).is_err()
        );
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
        let err = resolve_model_folder(vuho_model_paths::manifest().stt.default_model.as_str())
            .expect_err("nonexistent folder must not resolve");
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

    fn default_model_id() -> &'static str {
        vuho_model_paths::manifest().stt.default_model.as_str()
    }

    /// Lay out `components` under a fresh `label` temp directory: a
    /// `.mlmodelc` component is a directory, anything else a file.
    fn lay_out_components(label: &str, components: &[&str]) -> PathBuf {
        let tmp = std::env::temp_dir().join(label);
        fs::remove_dir_all(&tmp).ok();
        fs::create_dir_all(&tmp).expect("create tempdir");
        for comp in components {
            let p = tmp.join(comp);
            if comp.ends_with(".mlmodelc") {
                fs::create_dir_all(&p).expect("create component dir");
            } else {
                fs::write(&p, "").expect("create component file");
            }
        }
        tmp
    }

    /// A directory with all required components passes validation.
    #[test]
    fn validate_model_layout_passes_when_all_components_present() {
        let components = required_components(default_model_id()).expect("default model is known");
        let tmp = lay_out_components("vuho-test-model-ok", &components);

        assert!(validate_model_layout(default_model_id(), &tmp).is_ok());

        fs::remove_dir_all(&tmp).ok();
    }

    /// Validation fails when any single component is absent, and the error
    /// names that component rather than a generic "layout invalid".
    #[test]
    fn validate_model_layout_fails_naming_the_missing_component() {
        let mut components =
            required_components(default_model_id()).expect("default model is known");
        let absent = components.pop().expect("the model declares assets");
        let tmp = lay_out_components("vuho-test-model-missing", &components);

        let err =
            validate_model_layout(default_model_id(), &tmp).expect_err("a gap must be rejected");
        match err {
            EngineError::LoadFailed(msg) => {
                assert!(
                    msg.contains(absent),
                    "error should name {absent}, got: {msg}"
                );
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
        fs::remove_dir_all(&tmp).ok();
    }

    /// An empty directory fails on the very first required component.
    #[test]
    fn validate_model_layout_fails_on_empty_dir() {
        let components = required_components(default_model_id()).expect("default model is known");
        let tmp = lay_out_components("vuho-test-model-empty", &[]);

        let err =
            validate_model_layout(default_model_id(), &tmp).expect_err("an empty dir must fail");
        match err {
            EngineError::LoadFailed(msg) => {
                assert!(
                    msg.contains(components[0]),
                    "should name the first component {}, got: {msg}",
                    components[0]
                );
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
        fs::remove_dir_all(&tmp).ok();
    }

    /// An id the manifest doesn't know must be a typed error, never a
    /// silent fall-back to the default model.
    #[test]
    fn an_unknown_model_id_is_rejected_by_both_entry_points() {
        let unknown = "no-such-model";
        assert!(matches!(
            resolve_model_folder(unknown),
            Err(EngineError::UnknownModel(id)) if id == unknown
        ));
        assert!(matches!(
            validate_model_layout(unknown, Path::new("/nonexistent")),
            Err(EngineError::UnknownModel(id)) if id == unknown
        ));
    }
}
