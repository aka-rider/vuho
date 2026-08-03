//! Parakeet-TDT transcription engine.
//!
//! App-scoped, loaded once at startup (CONSTITUTION rule 3). Wraps
//! `Arc<SendModel<ParakeetModels>>` for thread-safe access — predictions are
//! still serialized onto one thread per session (see `coreml::SendModel`).
//!
//! Batch `transcribe` uses the sliding window + merge pipeline.
//! `start_stream` spawns the `"vuho-stt-session"` thread (`stream::session`)
//! and `stop_stream` joins it; `session` holds at most one live
//! [`SessionHandle`] at a time (CONSTITUTION rule 1: one owner per
//! resource — never a mirrored pair of fields that can disagree).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crossbeam_channel::Receiver;
use vuho_domain::{DictationEvent, TranscriptionResult};

use crate::coreml::SendModel;
use crate::EngineError;

use super::parakeet::decoder_state::DecoderState;
use super::parakeet::models::ParakeetModels;
use super::stream::accumulator::Accumulator;
use super::stream::merge;
use super::stream::session::{run_session, PARTIAL_INTERVAL};
use super::stream::windower;

/// Sentinel for "language detection produced nothing" — ISO 639-2's code
/// for "undetermined". CONSTITUTION rule 2: the producer's fact (here, "no
/// language was detected") must cross the boundary as data, never be
/// fabricated into a specific language downstream. Stamping `"en"` here
/// used to make `vuho-postprocess` apply English filler-word rules to
/// transcripts whose real language was never actually determined —
/// `vuho-postprocess::postprocess`'s own doc comment describes how it
/// treats this sentinel (skip language-specific filler removal, still
/// normalize generic formatting).
const UNKNOWN_LANGUAGE: &str = "und";

/// Resolve the language to stamp on a `TranscriptionResult`: whatever the
/// caller detected, or [`UNKNOWN_LANGUAGE`] if detection produced nothing —
/// never a guessed language code.
fn resolve_language(language: Option<&str>) -> String {
    language.unwrap_or(UNKNOWN_LANGUAGE).to_string()
}

/// A live streaming session's stop signal and join handle.
///
/// CONSTITUTION rule 9: the stopper (`ParakeetEngine::stop_stream`) owns
/// both — it sets `stop` and joins `join`, so there is never a stop flag
/// nobody reaches or a join nobody performs.
struct SessionHandle {
    stop: Arc<std::sync::atomic::AtomicBool>,
    join: std::thread::JoinHandle<TranscriptionResult>,
}

/// The Parakeet-TDT STT engine, loaded once at app startup.
pub struct ParakeetEngine {
    models: Arc<SendModel<ParakeetModels>>,
    /// At most one live streaming session (CONSTITUTION rule 1: a single
    /// `Option`, not a pair of fields that can disagree about whether a
    /// stream is active).
    session: Mutex<Option<SessionHandle>>,
}

impl ParakeetEngine {
    /// Load the Parakeet-TDT engine from a resolved model folder.
    ///
    /// The folder must contain all required `.mlmodelc` bundles and the
    /// vocabulary JSON (validated by [`crate::validate_model_layout`]
    /// during load). This also warms the models on the calling thread —
    /// callers that must not block should do this warmup off the command
    /// thread (as `vuho-ui`'s `spawn_warmup_and_bridge` does).
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if any model fails to load.
    ///
    /// Takes `PathBuf` by value (not `&Path`) so this composes directly with
    /// `resolve_model_folder().and_then(ParakeetEngine::load)` at call
    /// sites (`vuho-ui`'s `spawn_warmup_and_bridge`, `test-stt-ffi`).
    #[allow(clippy::needless_pass_by_value)]
    pub fn load(model_folder: PathBuf) -> Result<Self, EngineError> {
        let models = ParakeetModels::load(&model_folder)?;
        Ok(Self {
            models: Arc::new(SendModel(models)),
            session: Mutex::new(None),
        })
    }

    /// Stop and join whatever session is currently held.
    ///
    /// `None` means no session was active. `Some(Err(..))` means one was
    /// active but its thread panicked — distinct from "no session", so
    /// callers don't mistake a crashed session for a no-op stop.
    ///
    /// Shared by `stop_stream` (returns the result) and `unload` (logs and
    /// discards it) — one place performs the stop-flag-then-join sequence
    /// (CONSTITUTION rule 26: one source of truth per algorithm).
    fn take_and_stop_session(&self) -> Option<Result<TranscriptionResult, EngineError>> {
        let handle = {
            let mut guard = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.take()
        }?;
        handle.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        Some(handle.join.join().map_err(|_| EngineError::SessionPanicked))
    }
}

impl crate::TranscriptionEngine for ParakeetEngine {
    fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, EngineError> {
        let models = &self.models.0;
        let windows = windower::plan(samples.len());

        // Decoder state does NOT thread across `windower::plan` windows —
        // each window gets a *fresh* `DecoderState::new()` (blank/SOS
        // prime, zeroed h/c), not the previous window's carried-over LSTM
        // state.
        //
        // This deviates from the plan's original normative algorithm ("h =
        // zeros... once per session, not per window") deliberately. The
        // root cause of the multi-window content-drop bug (see
        // `tests/batch_multiwindow.rs`) is exactly a
        // carried-over decoder state: a window primed with LSTM state from
        // the *previous* window's last mid-sentence emission encodes a
        // strong "what comes next" expectation, but the new window's own,
        // independently-computed encoder output presents that same
        // acoustic content again at local frame 0 (the two windows overlap
        // in *audio*, not in a shared frame-index space) — a mismatch that
        // biases the joint toward blank at literally every frame (blank
        // never updates state, so a bad prime never self-corrects) until,
        // in the worst case, a strong-enough acoustic cue several seconds
        // later finally overrides it, silently dropping everything decoded
        // in between.
        //
        // FluidAudio sidesteps this class of bug architecturally, not with
        // a threshold: its `ChunkProcessor.swift` decodes each chunk from a
        // fresh `TdtDecoderState` (state cannot even be shared across the
        // parallel worker tasks that decode adjacent chunks). Cross-chunk
        // continuity comes entirely from the audio-level overlap (each
        // chunk independently re-decodes the tail of the previous chunk's
        // audio, now unbiased) plus token/word-level merge/dedup at the
        // seam — the `stream::merge` step below.
        //
        // `initial_t` is always 0 (not a carried `time_jump`): with a
        // fresh prime, a window independently re-decodes its own overlap
        // region from local frame 0 rather than skipping past it —
        // skipping would throw away the redundancy `merge` relies on.
        //
        // Word-granularity matching (not raw token id) and the resulting
        // seam re-splicing are `stream::merge::merge`'s job; folding the
        // outcome into `committed`/`segments` — including the
        // segments-vs-full_text truncation trade-off that re-splicing
        // implies — is `Accumulator::apply`'s (see its doc comment).
        let mut acc = Accumulator::new();

        for window in &windows {
            let window_samples = &samples[window.start..window.start + window.len];
            let mut state = DecoderState::new();
            let emitted =
                models.infer_window(window_samples, 0, window.global_frame_offset, &mut state)?;

            let outcome = merge::merge(acc.committed(), emitted, windower::OVERLAP_FRAMES, |id| {
                models.piece_info(id)
            });
            acc.apply(outcome, models);
        }

        let full_text = acc.full_text(models);
        Ok(TranscriptionResult {
            segments: acc.into_segments(),
            full_text,
            language: resolve_language(language),
        })
    }

    fn unload(&self) {
        // Stop any live session so its capture thread and CoreML calls
        // don't outlive the engine — then drop the last
        // `Arc<SendModel<ParakeetModels>>` reference (no explicit teardown
        // call exists on the CoreML side; releasing every reference is
        // teardown).
        if let Some(Err(e)) = self.take_and_stop_session() {
            log::warn!("parakeet: unload: streaming session stop failed: {e}");
        }
    }

    /// Returns an error if a session is already active — the engine enforces
    /// the trait's single-stream contract rather than silently replacing (and
    /// thereby leaking) a live session.
    fn start_stream(
        &self,
        language: Option<&str>,
        input_device: Option<&str>,
    ) -> Result<Receiver<DictationEvent>, EngineError> {
        // Precheck: a session is already active.
        {
            let guard = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if guard.is_some() {
                return Err(EngineError::StreamAlreadyActive);
            }
        }

        // Synchronous precheck: a known-denied/restricted status fails
        // immediately, without spawning a capture thread that would only
        // fail moments later. `NotDetermined` proceeds — macOS raises the
        // TCC dialog itself on the first real capture attempt inside
        // `vuho_audio::start_capture` (see `vuho-ui`'s
        // `request_mic_permission_on_startup` doc comment).
        match vuho_audio::mic_authorization_status() {
            vuho_audio::MicAuthStatus::Denied | vuho_audio::MicAuthStatus::Restricted => {
                return Err(EngineError::MicPermissionDenied);
            }
            vuho_audio::MicAuthStatus::Authorized | vuho_audio::MicAuthStatus::NotDetermined => {}
        }

        let capture_cfg = vuho_audio::CaptureConfig {
            device_name: input_device.map(str::to_string),
        };
        let (capture, chunk_rx) = vuho_audio::start_capture(&capture_cfg).map_err(|e| match e {
            vuho_audio::AudioError::PermissionDenied => EngineError::MicPermissionDenied,
            other => EngineError::Audio(other),
        })?;

        let (events_tx, events_rx) = crossbeam_channel::unbounded();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);
        let models = Arc::clone(&self.models);
        let language = resolve_language(language);

        let join = std::thread::Builder::new()
            .name("vuho-stt-session".into())
            .spawn(move || {
                // Rebind first: Rust 2021 disjoint closure capture would
                // otherwise capture only the `.0` field below, losing
                // `SendModel`'s `unsafe impl Send` (it applies to the whole
                // newtype, not its field).
                let models = models;
                run_session(
                    &chunk_rx,
                    &events_tx,
                    &stop_for_thread,
                    &models.0,
                    capture,
                    &language,
                    PARTIAL_INTERVAL,
                )
            })
            .map_err(|e| EngineError::SpawnFailed(e.to_string()))?;

        // Re-check for race: if another thread called start_stream concurrently,
        // stop this one's thread and return the "already active" error.
        let mut guard = self
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.is_some() {
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
            drop(join.join());
            return Err(EngineError::StreamAlreadyActive);
        }
        *guard = Some(SessionHandle { stop, join });
        Ok(events_rx)
    }

    fn stop_stream(&self) -> Result<TranscriptionResult, EngineError> {
        self.take_and_stop_session()
            .unwrap_or(Err(EngineError::NoActiveStream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CONSTITUTION rule 2: an undetected language must surface as the
    /// honest "undetermined" sentinel, never a fabricated guess like
    /// `"en"` — model-free, so this runs without the `CoreML` model loaded.
    #[test]
    fn resolve_language_falls_back_to_the_undetermined_sentinel_not_english() {
        assert_eq!(resolve_language(None), "und");
    }

    #[test]
    fn resolve_language_passes_through_a_detected_language_unchanged() {
        assert_eq!(resolve_language(Some("vi")), "vi");
        assert_eq!(resolve_language(Some("en")), "en");
    }
}
