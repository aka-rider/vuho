//! The backend-independent half of every engine: batch windowing, the
//! streaming session lifecycle, and the single live-session slot.
//!
//! Everything here is generic over a [`WindowInference`] backend, so
//! `ParakeetEngine` and `CanaryEngine` are thin wrappers rather than two
//! copies of the same lifecycle (CONSTITUTION rule 26). The backend only
//! supplies one-window decoding, a vocabulary, and its merge bounds.
//!
//! `where SendModel<M>: Send + Sync` appears on the impl block by
//! necessity, not by style: `coreml::SendModel`'s `unsafe impl`s are
//! deliberately written per concrete type and a blanket `impl<T>` is
//! refused there, so a generic wrapper cannot derive thread-safety — it
//! has to demand the concrete impl the backend's own module provides.

use std::sync::{Arc, Mutex};

use crossbeam_channel::Receiver;
use vuho_domain::{DictationEvent, TranscriptionResult};

use crate::coreml::SendModel;
use crate::stream::accumulator::Accumulator;
use crate::stream::session::{run_session, PARTIAL_INTERVAL};
use crate::stream::{merge, windower};
use crate::window_inference::WindowInference;
use crate::EngineError;

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
pub(crate) fn resolve_language(language: Option<&str>) -> String {
    language.unwrap_or(UNKNOWN_LANGUAGE).to_string()
}

/// A live streaming session's stop signal and join handle.
///
/// CONSTITUTION rule 9: the stopper ([`StreamingEngine::stop_stream`]) owns
/// both — it sets `stop` and joins `join`, so there is never a stop flag
/// nobody reaches or a join nobody performs.
struct SessionHandle {
    stop: Arc<std::sync::atomic::AtomicBool>,
    join: std::thread::JoinHandle<TranscriptionResult>,
}

/// A loaded backend plus at most one live streaming session.
pub(crate) struct StreamingEngine<M> {
    models: Arc<SendModel<M>>,
    /// At most one live streaming session (CONSTITUTION rule 1: a single
    /// `Option`, not a pair of fields that can disagree about whether a
    /// stream is active).
    session: Mutex<Option<SessionHandle>>,
}

impl<M: WindowInference + 'static> StreamingEngine<M>
where
    SendModel<M>: Send + Sync,
{
    pub(crate) fn new(models: M) -> Self {
        Self {
            models: Arc::new(SendModel(models)),
            session: Mutex::new(None),
        }
    }

    /// Batch transcription over the sliding-window plan.
    ///
    /// Decoder state does NOT thread across `windower::plan` windows — each
    /// window decodes from scratch, which is [`WindowInference::infer_window`]'s
    /// contract (ADR-015). The root cause of the multi-window content-drop
    /// bug (see `tests/batch_multiwindow.rs`) was exactly a carried-over
    /// decoder state: a window primed with state from the *previous*
    /// window's last mid-sentence emission encodes a strong "what comes
    /// next" expectation, but the new window's own, independently-computed
    /// encoder output presents that same acoustic content again at local
    /// frame 0 — a mismatch that biases the decode toward blank at every
    /// frame until, in the worst case, everything in between is silently
    /// dropped.
    ///
    /// Cross-window continuity therefore comes entirely from the audio-level
    /// overlap plus word-level reconciliation at the seam
    /// (`stream::merge::merge`, with the backend's own
    /// [`WindowInference::merge_bounds`]); folding the outcome into
    /// `committed`/`segments` is `Accumulator::apply`'s job.
    ///
    /// # Errors
    ///
    /// Propagates whatever [`WindowInference::infer_window`] returns.
    pub(crate) fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, EngineError> {
        let models = &self.models.0;
        let session_language = resolve_language(language);
        let mut acc = Accumulator::new();

        for window in &windower::plan(samples.len()) {
            let window_samples = &samples[window.start..window.start + window.len];
            let emitted = models.infer_window(
                window_samples,
                window.global_frame_offset,
                &session_language,
            )?;

            let outcome = merge::merge(acc.committed(), emitted, models.merge_bounds(), |id| {
                models.piece_info(id)
            });
            acc.apply(outcome, models);
        }

        let full_text = acc.full_text(models);
        Ok(TranscriptionResult {
            segments: acc.into_segments(),
            full_text,
            language: session_language,
        })
    }

    /// Stop and join whatever session is currently held.
    ///
    /// `None` means no session was active. `Some(Err(..))` means one was
    /// active but its thread panicked — distinct from "no session", so
    /// callers don't mistake a crashed session for a no-op stop.
    ///
    /// Shared by [`Self::stop_stream`] (returns the result) and
    /// [`Self::unload`] (logs and discards it) — one place performs the
    /// stop-flag-then-join sequence (CONSTITUTION rule 26).
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

    /// Stop any live session so its capture thread and `CoreML` calls don't
    /// outlive the engine — then drop the last `Arc<SendModel<M>>`
    /// reference (no explicit teardown call exists on the `CoreML` side;
    /// releasing every reference is teardown).
    pub(crate) fn unload(&self) {
        if let Some(Err(e)) = self.take_and_stop_session() {
            log::warn!("streaming engine: unload: streaming session stop failed: {e}");
        }
    }

    /// Start a streaming session.
    ///
    /// Returns an error if a session is already active — the engine enforces
    /// the trait's single-stream contract rather than silently replacing (and
    /// thereby leaking) a live session.
    ///
    /// # Errors
    ///
    /// See [`crate::TranscriptionEngine::start_stream`].
    pub(crate) fn start_stream(
        &self,
        language: Option<&str>,
        input_device: Option<&str>,
    ) -> Result<Receiver<DictationEvent>, EngineError> {
        self.ensure_no_active_session()?;
        Self::ensure_mic_not_denied()?;

        let capture_cfg = vuho_audio::CaptureConfig {
            device_name: input_device.map(str::to_string),
        };
        let (capture, chunk_rx) = vuho_audio::start_capture(&capture_cfg).map_err(|e| match e {
            vuho_audio::AudioError::PermissionDenied => EngineError::MicPermissionDenied,
            other => EngineError::Audio(other),
        })?;

        let (events_tx, events_rx) = crossbeam_channel::unbounded();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let join = self.spawn_session(
            chunk_rx,
            events_tx,
            Arc::clone(&stop),
            capture,
            resolve_language(language),
        )?;

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

    /// Spawn the `"vuho-stt-session"` thread that owns the decode loop.
    fn spawn_session(
        &self,
        chunks: Receiver<Vec<f32>>,
        events: crossbeam_channel::Sender<DictationEvent>,
        stop: Arc<std::sync::atomic::AtomicBool>,
        capture: vuho_audio::CaptureHandle,
        language: String,
    ) -> Result<std::thread::JoinHandle<TranscriptionResult>, EngineError> {
        let models = Arc::clone(&self.models);
        std::thread::Builder::new()
            .name("vuho-stt-session".into())
            .spawn(move || {
                // Rebind first: Rust 2021 disjoint closure capture would
                // otherwise capture only the `.0` field below, losing
                // `SendModel`'s `unsafe impl Send` (it applies to the whole
                // newtype, not its field).
                let models = models;
                run_session(
                    &chunks,
                    &events,
                    &stop,
                    &models.0,
                    capture,
                    &language,
                    PARTIAL_INTERVAL,
                )
            })
            .map_err(|e| EngineError::SpawnFailed(e.to_string()))
    }

    fn ensure_no_active_session(&self) -> Result<(), EngineError> {
        let guard = self
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.is_some() {
            return Err(EngineError::StreamAlreadyActive);
        }
        Ok(())
    }

    /// Synchronous precheck: a known-denied/restricted status fails
    /// immediately, without spawning a capture thread that would only fail
    /// moments later. `NotDetermined` proceeds — macOS raises the TCC dialog
    /// itself on the first real capture attempt inside
    /// `vuho_audio::start_capture` (see `vuho-ui`'s
    /// `request_mic_permission_on_startup` doc comment).
    fn ensure_mic_not_denied() -> Result<(), EngineError> {
        match vuho_audio::mic_authorization_status() {
            vuho_audio::MicAuthStatus::Denied | vuho_audio::MicAuthStatus::Restricted => {
                Err(EngineError::MicPermissionDenied)
            }
            vuho_audio::MicAuthStatus::Authorized | vuho_audio::MicAuthStatus::NotDetermined => {
                Ok(())
            }
        }
    }

    /// Stop the active streaming session and return the final transcription.
    ///
    /// # Errors
    ///
    /// See [`crate::TranscriptionEngine::stop_stream`].
    pub(crate) fn stop_stream(&self) -> Result<TranscriptionResult, EngineError> {
        self.take_and_stop_session()
            .unwrap_or(Err(EngineError::NoActiveStream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CONSTITUTION rule 2: an undetected language must surface as the
    /// honest "undetermined" sentinel, never a fabricated guess like
    /// `"en"` — model-free, so this runs without any model loaded.
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
