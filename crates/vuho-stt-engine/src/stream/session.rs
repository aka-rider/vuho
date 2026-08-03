//! The streaming session thread (`"vuho-stt-session"`).
//!
//! Consumes 16 kHz mono audio chunks from `vuho-audio`, runs VAD-gated
//! partial re-inference of the open sliding window, commits full windows,
//! and emits `PartialTranscript`/`Activity`/`Error` events — never
//! `SessionStarted`/`SessionCompleted`, which are the pipeline's job
//! (CONSTITUTION rule 11: only the caller that actually started the
//! session knows it succeeded).
//!
//! [`run_session`] is a free function over a chunk [`Receiver`] + an
//! events [`Sender`] (CONSTITUTION rule 5-adjacent: no `cpal` dependency
//! here) so it is unit-testable without a real microphone — see the
//! `AudioSource` trait below and this module's tests.
//!
//! The committed-transcript bookkeeping (`committed` tokens, `segments`,
//! `segment_id`) is not kept here — [`SessionState`] holds a
//! `stream::accumulator::Accumulator`, the one place that logic lives,
//! shared with batch `transcribe()` (`engine.rs`).
//!
//! # Decoder state is never carried across inferences
//!
//! Every inference in this module — a partial re-inference of the open
//! window, a full window commit, VAD-endpoint promotion, and the final
//! end-aligned tail — decodes from a **fresh** `DecoderState::new()` and
//! `initial_t = 0`. Nothing here threads decoder state from one inference
//! into the next.
//!
//! This mirrors `engine.rs`'s batch `transcribe()` (see its doc comment
//! for the full root-cause writeup) and `FluidAudio`'s own
//! `ChunkProcessor.swift`, which decodes each chunk from a fresh
//! `TdtDecoderState` — state literally cannot be shared across the
//! parallel worker tasks that decode adjacent chunks there. Carrying
//! decoder state between two independently-computed encoder outputs (each
//! inference here re-encodes the growing/committed buffer from scratch) is
//! exactly what caused the characterized blank-lock content-drop bug: a
//! decoder primed with LSTM state from a previous inference's last
//! mid-sentence emission encodes a strong "what comes next" expectation,
//! but the new inference's own encoder output presents that same acoustic
//! content again at local frame 0 — a mismatch that can bias the joint
//! toward blank at every frame (blank never updates state, so a bad prime
//! never self-corrects). Always starting fresh sidesteps this entirely;
//! reconciling the resulting overlap between independently-correct
//! decodes is `merge`'s job (word-granularity matching — see
//! `stream::merge`'s doc comment — since two fresh decodes can split the
//! same word into different subword pieces).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use vuho_domain::{DictationEvent, ErrorKind, TranscriptionResult};

use crate::parakeet::decoder_state::DecoderState;
use crate::parakeet::models::ParakeetModels;
use crate::parakeet::tdt::TokenAt;
use crate::stream::accumulator::Accumulator;
use crate::stream::merge::MergeOutcome;
use crate::stream::{merge, windower};
use crate::EngineError;

/// How often the recv loop times out to re-check the stop flag when no
/// chunk has arrived, and (in that same idle branch, see [`run_session`])
/// the ceiling on the cadence-driven partial re-inference check.
///
/// This is *not* the `Activity` event cadence while audio is flowing, only
/// an upper bound on it while idle: `Activity` is sent once per loop
/// iteration (`run_session`), and a busy loop iterates once per *received*
/// chunk — i.e. at `vuho-audio`'s pump cadence (`capture.rs`'s
/// `PUMP_POLL`, ~10 ms), roughly 5x faster than this 50 ms constant. That
/// gap is intentional, not something to rate-limit away: each `Activity`
/// send is one cheap push onto an unbounded channel, and nothing downstream
/// does more work per event than it would per frame — throttling sends to
/// match this constant would add state and complexity for no measurable
/// benefit.
const RECV_TIMEOUT: Duration = Duration::from_millis(50);
/// Minimum wall-clock interval between partial re-inferences of the open
/// window, in production. "Adaptive: only after the previous inference
/// returned" holds automatically here — this is a single-threaded loop, so
/// a new partial can never start while a previous one is still running.
///
/// Injectable: [`run_session`] takes this as a parameter (`partial_interval`)
/// rather than using this constant directly, so tests can pass
/// `Duration::ZERO` to decouple event ordering from wall-clock pacing. Both
/// production call sites (`ParakeetEngine::start_stream`) pass this
/// constant.
pub(crate) const PARTIAL_INTERVAL: Duration = Duration::from_secs(1);
/// Trailing silence (ms) that promotes the open window's unconfirmed
/// tokens to committed, without advancing the window.
const ENDPOINT_SILENCE_MS: u32 = 800;

/// A source of live audio: capture-level RMS for the cosmetic waveform,
/// plus a one-shot stop. Implemented by [`vuho_audio::CaptureHandle`] in
/// production and by a trivial fake in tests, so [`run_session`] never
/// depends on `cpal`.
pub(crate) trait AudioSource {
    /// Root-mean-square level of the most recently processed audio block.
    fn level_rms(&self) -> f32;
    /// Stop the underlying capture and release its thread. Called at most
    /// once, when the session observes the stop flag.
    fn stop(self);
    /// Take (clear) any error the capture thread recorded. Consulted only
    /// when the chunk channel disconnects *without* a requested stop (see
    /// [`handle_disconnected`]), to surface the specific failure (e.g.
    /// `StreamDied`) instead of a generic message — this is what makes
    /// [`vuho_audio::CaptureHandle::take_error`] (previously write-only:
    /// written by the capture thread, never read by any caller) live.
    fn take_error(&self) -> Option<vuho_audio::AudioError>;
}

impl AudioSource for vuho_audio::CaptureHandle {
    fn level_rms(&self) -> f32 {
        Self::level_rms(self)
    }

    fn stop(self) {
        Self::stop(self);
    }

    fn take_error(&self) -> Option<vuho_audio::AudioError> {
        Self::take_error(self)
    }
}

/// Map a capture RMS level to a normalized `[0.0, 1.0]` activity value.
///
/// `((20*log10(rms)+50)/50).clamp(0,1)` per the plan: roughly maps -50 dBFS
/// (near silence) to 0.0 and 0 dBFS (full scale) to 1.0. `log10(0.0)` is
/// `-inf`, which `clamp` reduces to `0.0` without panicking.
fn activity_level(rms: f32) -> f32 {
    ((20.0 * rms.log10() + 50.0) / 50.0).clamp(0.0, 1.0)
}

/// Mutable state threaded across the whole session: the currently open
/// (not yet committed) window's buffered audio, the committed transcript
/// (via [`Accumulator`]), and the outstanding unconfirmed ("fresh") tokens
/// from the last partial re-inference.
///
/// No decoder state lives here — see this module's doc comment. Every
/// inference builds its own `DecoderState::new()` on the spot.
struct SessionState {
    /// Samples accumulated for the currently open (not yet committed) window.
    open_buffer: Vec<f32>,
    /// Sample offset of `open_buffer[0]` within the whole session's audio.
    window_base_offset: usize,
    /// The confirmed transcript: committed tokens + segments (shared home
    /// with batch `transcribe()` — see `stream::accumulator`).
    acc: Accumulator,
    /// Unconfirmed tokens from the last partial re-inference of the open
    /// window.
    fresh: Vec<TokenAt>,
    /// How much of `acc.committed()` to keep (`Vec::truncate`) if `fresh`
    /// is promoted — the other half of the last partial's `MergeOutcome`
    /// (see `run_partial`'s doc comment). `merge::merge` can decide to
    /// re-splice a seam using `fresh`'s copy of some already-committed
    /// words (see `stream::merge`'s doc comment), so `fresh` alone is not
    /// enough to promote correctly: promoting it as a blind `extend`
    /// without first truncating `committed` back to this value would
    /// duplicate exactly those re-spliced seam words. Meaningless (never
    /// read) whenever `fresh` is empty, which is the only state it starts
    /// in and returns to after every promotion/commit.
    fresh_keep_committed: usize,
}

impl SessionState {
    fn new() -> Self {
        Self {
            open_buffer: Vec::with_capacity(windower::WINDOW_SAMPLES),
            window_base_offset: 0,
            acc: Accumulator::new(),
            fresh: Vec::new(),
            fresh_keep_committed: 0,
        }
    }

    fn global_frame_offset(&self) -> usize {
        self.window_base_offset / windower::SAMPLES_PER_FRAME
    }

    /// `(confirmed_text, unconfirmed_text)` for a `PartialTranscript` event.
    fn transcript_texts(&self, models: &ParakeetModels) -> (String, String) {
        self.acc.confirmed_unconfirmed_texts(&self.fresh, models)
    }
}

/// Tracks whether enough time has passed, and there has been speech, since
/// the last partial re-inference to run another one — the single place
/// that cadence bookkeeping (as opposed to the pure decision in
/// [`due_for_partial`]) lives.
struct PartialCadence {
    interval: Duration,
    last_ran_at: Option<Instant>,
    speech_since_last: bool,
    /// One-shot latch for the VAD-endpoint flush (see [`endpoint_flush_due`]):
    /// `true` once a flush has run since the last speech (or no speech has
    /// happened yet — nothing to flush), `false` from `note_speech()` until
    /// the next `mark_flushed()`. Lives here, not a second struct, because
    /// it is exactly the same "has X happened since speech" shape this
    /// struct already tracks for the cadence-driven partial.
    flushed_since_speech: bool,
}

impl PartialCadence {
    fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_ran_at: None,
            speech_since_last: false,
            flushed_since_speech: true,
        }
    }

    fn note_speech(&mut self) {
        self.speech_since_last = true;
        self.flushed_since_speech = false;
    }

    fn due(&self) -> bool {
        due_for_partial(self.last_ran_at, self.speech_since_last, self.interval)
    }

    fn mark_ran(&mut self) {
        self.last_ran_at = Some(Instant::now());
        self.speech_since_last = false;
    }

    fn flushed_since_speech(&self) -> bool {
        self.flushed_since_speech
    }

    fn mark_flushed(&mut self) {
        self.flushed_since_speech = true;
    }
}

/// Whether enough time has passed since the last partial re-inference, and
/// there has been speech since then, to run another one.
fn due_for_partial(
    last_ran_at: Option<Instant>,
    speech_since_last: bool,
    interval: Duration,
) -> bool {
    speech_since_last && last_ran_at.is_none_or(|t| t.elapsed() >= interval)
}

/// A silence endpoint is due for a one-shot flush re-inference when trailing
/// silence has crossed the endpoint threshold and no flush has run since the
/// last speech.
///
/// The one-shot latch (`flushed_since_speech`) is what keeps this to exactly
/// one extra inference per pause: without it, every chunk received while
/// still silent would trigger another full-window re-decode.
fn endpoint_flush_due(trailing_silence_ms: u32, flushed_since_speech: bool) -> bool {
    !flushed_since_speech && trailing_silence_ms >= ENDPOINT_SILENCE_MS
}

/// Merge a fresh partial re-decode (`emitted`) against `state.acc`'s
/// committed tokens and stash the resulting [`MergeOutcome`] — both halves
/// — into `state.fresh`/`state.fresh_keep_committed`.
///
/// Pure aside from the `merge::merge` call (no `CoreML`, no event send): split
/// out of [`run_partial`] so the merge/stash step is unit-testable without a
/// real model beyond the `piece` lookup `merge::merge` itself needs (see
/// this module's tests). A partial never touches `committed` itself here —
/// only a commit ([`commit_window`]) or a VAD promotion
/// ([`promote_fresh_to_committed`]) does.
fn apply_partial_merge<'p>(
    state: &mut SessionState,
    emitted: Vec<TokenAt>,
    piece: impl Fn(u32) -> Option<(bool, &'p str)>,
) {
    let merged = merge::merge(
        state.acc.committed(),
        emitted,
        windower::OVERLAP_FRAMES,
        piece,
    );
    // `fresh` is the append side of the outcome, displayed as unconfirmed
    // text. `keep_committed` is stashed alongside it: if this partial's
    // tokens are later promoted (VAD endpoint), the promotion must
    // truncate `committed` to `keep_committed` first (see
    // `SessionState::fresh_keep_committed`'s doc comment) — dropping it
    // here, keeping only `append`, is what let a promoted seam duplicate
    // words already in `committed`.
    state.fresh_keep_committed = merged.keep_committed;
    state.fresh = merged.append;
}

/// Run one partial re-inference of the open window from a fresh decoder
/// state, updating `state.fresh`, and emit a `PartialTranscript`.
///
/// Re-decodes the *whole* open buffer from scratch every time (see this
/// module's doc comment) — deterministic given the same buffer contents,
/// so re-running is naturally idempotent.
fn run_partial(state: &mut SessionState, models: &ParakeetModels, events: &Sender<DictationEvent>) {
    let global_frame_offset = state.global_frame_offset();
    let mut fresh_state = DecoderState::new();
    let started_at = Instant::now();
    let outcome = models.infer_window(&state.open_buffer, 0, global_frame_offset, &mut fresh_state);
    log::debug!(
        "vuho-stt-session: partial inference took {:?} over open_buffer.len()={}",
        started_at.elapsed(),
        state.open_buffer.len()
    );
    let Ok(emitted) = outcome else {
        let e = outcome.unwrap_err();
        log::warn!("vuho-stt-session: partial inference failed: {e}");
        report_inference_failure(events, "partial", &e);
        return;
    };

    apply_partial_merge(state, emitted, |id| models.piece_info(id));

    let (confirmed_text, unconfirmed_text) = state.transcript_texts(models);
    send(
        events,
        DictationEvent::PartialTranscript {
            confirmed_text,
            unconfirmed_text,
        },
    );
}

/// VAD endpoint: promote the open window's unconfirmed tokens to
/// committed, without advancing the window. This only affects which
/// tokens count as "confirmed" for the UI split and as the `merge` anchor
/// for the next partial.
///
/// Applies the *whole* `MergeOutcome` the last `run_partial` computed —
/// `keep_committed` (via `state.fresh_keep_committed`) as well as `append`
/// (`state.fresh`) — through `Accumulator::apply`, exactly like a window
/// commit does. Promoting `fresh` alone (a blind `committed.extend`) would
/// re-add whatever seam words `merge` had already decided to keep only
/// `fresh`'s copy of, permanently duplicating them once VAD makes the
/// promotion stick.
fn promote_fresh_to_committed(
    state: &mut SessionState,
    models: &ParakeetModels,
    events: &Sender<DictationEvent>,
) {
    if state.fresh.is_empty() {
        return;
    }
    let promoted = std::mem::take(&mut state.fresh);
    let outcome = MergeOutcome {
        keep_committed: state.fresh_keep_committed,
        append: promoted,
    };
    state.acc.apply(outcome, models);

    let (confirmed_text, unconfirmed_text) = state.transcript_texts(models);
    send(
        events,
        DictationEvent::PartialTranscript {
            confirmed_text,
            unconfirmed_text,
        },
    );
}

/// Run one full inference over `samples` (fresh decoder state, per this
/// module's doc comment), merge it against `acc`'s committed tokens, and
/// fold the outcome in.
///
/// Takes `acc: &mut Accumulator` (not `state: &mut SessionState`) so a
/// caller can pass `&mut state.acc` alongside a borrowed slice of
/// `state.open_buffer` — a disjoint field borrow the compiler accepts, a
/// whole-`&mut SessionState` parameter would not (WP9: this is what lets
/// [`commit_window`] infer directly over a slice of `open_buffer` instead
/// of first cloning it into an owned `Vec`).
///
/// Shared by [`commit_window`] (a full window commit, which additionally
/// advances the window afterward) and [`finalize_tail`] (the end-aligned
/// final window, which doesn't) — CONSTITUTION rule 26: one source of
/// truth for the infer → merge → apply sequence.
///
/// On inference failure, the caller still advances/finishes afterward (see
/// [`commit_window`]/[`finalize_tail`]) rather than retrying: the same
/// samples against the same failing model call would just fail again,
/// looping forever instead of making progress, so the window's audio is
/// lost either way. [`report_inference_failure`] is what makes that loss
/// user-visible instead of a silent, only-in-the-log drop (CONSTITUTION
/// rule 10).
fn infer_and_apply(
    acc: &mut Accumulator,
    global_frame_offset: usize,
    models: &ParakeetModels,
    samples: &[f32],
    context: &str,
    events: &Sender<DictationEvent>,
) {
    let mut fresh_state = DecoderState::new();
    match models.infer_window(samples, 0, global_frame_offset, &mut fresh_state) {
        Ok(emitted) => {
            let outcome = merge::merge(acc.committed(), emitted, windower::OVERLAP_FRAMES, |id| {
                models.piece_info(id)
            });
            acc.apply(outcome, models);
        }
        Err(e) => {
            log::warn!("vuho-stt-session: {context} inference failed: {e}");
            report_inference_failure(events, context, &e);
        }
    }
}

/// Build and send the recoverable `Error` event for a failed commit/final-
/// window inference (see [`infer_and_apply`]'s doc comment for why the
/// caller still advances afterward). Split out from `infer_and_apply` so
/// it's unit-testable without a real `ParakeetModels`/`CoreML` call.
fn report_inference_failure(events: &Sender<DictationEvent>, context: &str, e: &EngineError) {
    send(
        events,
        DictationEvent::Error {
            message: format!("{context} inference failed: {e}"),
            recoverable: true,
            kind: ErrorKind::Other,
        },
    );
}

/// Window commit: the open window reached `WINDOW_SAMPLES`. Runs a full
/// inference over exactly `WINDOW_SAMPLES` samples, merges into committed,
/// and advances the window by `ADVANCE`.
fn commit_window(
    state: &mut SessionState,
    models: &ParakeetModels,
    events: &Sender<DictationEvent>,
) {
    let global_frame_offset = state.global_frame_offset();
    // `&mut state.acc` and `&state.open_buffer[..]` are disjoint field
    // borrows, so this infers directly over a slice of `open_buffer` —
    // no `to_vec()` clone of the (240 000-sample) window needed.
    infer_and_apply(
        &mut state.acc,
        global_frame_offset,
        models,
        &state.open_buffer[..windower::WINDOW_SAMPLES],
        "window commit",
        events,
    );

    // Advance: drop the first ADVANCE samples in place (WP9: was a
    // split_off + slice-to_vec + extend triple copy; `Vec::drain` of a
    // leading range memmoves the remaining tail down and truncates, one
    // pass, no extra allocation). What's left is exactly the last
    // OVERLAP_SAMPLES of the just-inferred window, plus whatever
    // overflowed past WINDOW_SAMPLES while we were still filling it —
    // both already sit after `ADVANCE` in the buffer, untouched.
    state.open_buffer.drain(..windower::ADVANCE);
    state.window_base_offset += windower::ADVANCE;
    state.fresh.clear();
    state.fresh_keep_committed = 0;

    let (confirmed_text, unconfirmed_text) = state.transcript_texts(models);
    send(
        events,
        DictationEvent::PartialTranscript {
            confirmed_text,
            unconfirmed_text,
        },
    );
}

/// End-aligned final window on stop: one last inference over whatever
/// remains in the open buffer, folded into committed.
fn finalize_tail(
    state: &mut SessionState,
    models: &ParakeetModels,
    events: &Sender<DictationEvent>,
) {
    if state.open_buffer.is_empty() {
        return;
    }
    let global_frame_offset = state.global_frame_offset();
    let tail = std::mem::take(&mut state.open_buffer);
    infer_and_apply(
        &mut state.acc,
        global_frame_offset,
        models,
        &tail,
        "final window",
        events,
    );
}

/// Send an event, logging (not panicking) if the receiver is gone —
/// CONSTITUTION rule 10: a closed channel is handled, never ignored.
fn send(events: &Sender<DictationEvent>, event: DictationEvent) {
    if events.send(event).is_err() {
        log::warn!("vuho-stt-session: event channel closed — consumer gone");
    }
}

/// Handle one received audio chunk: feed VAD, buffer it, promote to
/// committed on a VAD silence endpoint, then run whichever of window-commit
/// or cadence-driven partial re-inference is due.
///
/// `audio_live` mirrors the caller's `audio.is_some()` — once the stop path
/// has taken `audio`, the loop is only draining already-buffered chunks, so
/// cadence-driven partials are skipped (the drained tail goes straight to
/// `finalize_tail` once the channel disconnects).
fn handle_chunk(
    chunk: &[f32],
    vad: &mut crate::vad::Vad,
    state: &mut SessionState,
    models: &ParakeetModels,
    events: &Sender<DictationEvent>,
    audio_live: bool,
    cadence: &mut PartialCadence,
) {
    let vad_update = vad.push(chunk);
    if vad_update.any_speech {
        cadence.note_speech();
    }
    state.open_buffer.extend_from_slice(chunk);

    // Flush before promote: re-infer the open window once with the full
    // trailing silence as right-context (a greedy TDT decode routinely
    // withholds the last word(s) of an utterance until it sees enough
    // right-context), THEN promote that fresh result — never the stale
    // `state.fresh` from whatever partial happened to run mid-utterance.
    // The latch (`flushed_since_speech`) caps this to one extra inference
    // per pause; without it every chunk received while still silent would
    // re-trigger a full-window decode.
    if endpoint_flush_due(
        vad_update.trailing_silence_ms,
        cadence.flushed_since_speech(),
    ) {
        run_partial(state, models, events);
        cadence.mark_ran();
        cadence.mark_flushed();
        promote_fresh_to_committed(state, models, events);
    }

    // `while`, not `if`: a single received chunk can itself be larger than
    // `ADVANCE` (nothing upstream in `vuho-audio` bounds chunk size to
    // that), in which case one commit's `drain(..ADVANCE)` can still leave
    // `open_buffer.len() >= WINDOW_SAMPLES` — an `if` would leave the
    // window over-full, hitting `infer_window`'s `debug_assert!(samples.len()
    // <= window_samples)` in debug builds or silently truncating the excess
    // in release. Looping drains every full window a chunk completes before
    // falling through to the mutually-exclusive partial re-inference below.
    let mut committed_this_chunk = false;
    while state.open_buffer.len() >= windower::WINDOW_SAMPLES {
        commit_window(state, models, events);
        cadence.mark_ran();
        committed_this_chunk = true;
    }
    if !committed_this_chunk && audio_live && cadence.due() {
        run_partial(state, models, events);
        cadence.mark_ran();
    }
}

/// The chunk channel disconnected. `audio` distinguishes the normal end
/// (`None`: we asked to stop, `audio.take()` already ran, and the capture
/// thread fully flushed and dropped its sender — run the end-aligned final
/// window) from an unexpected capture death (`Some`: never asked to stop —
/// report a recoverable `Error` carrying the specific
/// [`vuho_audio::AudioError`] the capture thread recorded, if any, via
/// [`AudioSource::take_error`], CONSTITUTION rule 10, and skip the
/// final-window inference).
fn handle_disconnected<A: AudioSource>(
    state: &mut SessionState,
    models: &ParakeetModels,
    events: &Sender<DictationEvent>,
    audio: Option<&A>,
) {
    let Some(audio) = audio else {
        finalize_tail(state, models, events);
        return;
    };

    let detail = audio.take_error();
    log::error!("vuho-stt-session: capture channel disconnected unexpectedly (detail={detail:?})");
    let message = detail.map_or_else(
        || "audio capture ended unexpectedly".to_string(),
        |e| e.to_string(),
    );
    send(
        events,
        DictationEvent::Error {
            message,
            recoverable: true,
            kind: ErrorKind::Other,
        },
    );
}

/// Run the streaming session loop to completion, returning the assembled
/// `TranscriptionResult`.
///
/// `chunks` yields successive 16 kHz mono audio blocks (as produced by
/// `vuho_audio::start_capture`, or a test fake). `stop` is observed via
/// `recv_timeout`'s natural ~50 ms cadence; once set, `audio.stop()` is
/// called exactly once, after which the loop drains any remaining
/// buffered chunks until `chunks` disconnects (crossbeam channels yield
/// every buffered message before reporting `Disconnected`), then runs the
/// end-aligned final window and returns.
///
/// `partial_interval` is the minimum wall-clock gap between partial
/// re-inferences of the open window (production passes [`PARTIAL_INTERVAL`];
/// tests can pass `Duration::ZERO` to decouple event-ordering assertions
/// from wall-clock pacing).
pub(crate) fn run_session<A: AudioSource>(
    chunks: &Receiver<Vec<f32>>,
    events: &Sender<DictationEvent>,
    stop: &AtomicBool,
    models: &ParakeetModels,
    audio: A,
    language: &str,
    partial_interval: Duration,
) -> TranscriptionResult {
    let mut vad = match crate::vad::Vad::new() {
        Ok(v) => v,
        Err(e) => {
            send(
                events,
                DictationEvent::Error {
                    message: format!("VAD init failed: {e}"),
                    recoverable: false,
                    kind: ErrorKind::Other,
                },
            );
            return TranscriptionResult {
                segments: vec![],
                full_text: String::new(),
                language: language.to_string(),
            };
        }
    };

    let mut state = SessionState::new();
    let mut audio = Some(audio);
    let mut cadence = PartialCadence::new(partial_interval);

    loop {
        if stop.load(Ordering::SeqCst) {
            if let Some(a) = audio.take() {
                a.stop();
            }
        }
        if let Some(a) = audio.as_ref() {
            send(
                events,
                DictationEvent::Activity {
                    level: activity_level(a.level_rms()),
                },
            );
        }

        match chunks.recv_timeout(RECV_TIMEOUT) {
            Ok(chunk) => handle_chunk(
                &chunk,
                &mut vad,
                &mut state,
                models,
                events,
                audio.is_some(),
                &mut cadence,
            ),
            Err(RecvTimeoutError::Timeout) => {
                if audio.is_some() && cadence.due() {
                    run_partial(&mut state, models, events);
                    cadence.mark_ran();
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                handle_disconnected(&mut state, models, events, audio.as_ref());
                break;
            }
        }
    }

    let full_text = state.acc.full_text(models);
    TranscriptionResult {
        segments: state.acc.into_segments(),
        full_text,
        language: language.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    /// A test audio source: no real capture thread, and a level derived
    /// from nothing (the waveform is cosmetic and not under test here).
    ///
    /// `stop()` drops its held `Sender` clone — replicating production's
    /// invariant that only the session's *own* stop path closes the chunk
    /// channel (`CaptureHandle::stop` flushes and drops its sender as its
    /// last act). Without this, a test feeder thread that independently
    /// drops its sender clone right after setting the stop flag races the
    /// session's own `stop.load()` check: the channel can disconnect
    /// before the session ever observes the flag, misreporting a graceful
    /// stop as an unexpected capture death and skipping the final-window
    /// inference entirely.
    struct TestAudioSource {
        keep_alive: Sender<Vec<f32>>,
        /// Simulates `CaptureHandle::take_error`'s recorded error — `None`
        /// by default (matching most tests, which never simulate a real
        /// capture failure); tests that DO want to assert on the specific
        /// error surfaced by an unexpected disconnect set this.
        error: std::cell::Cell<Option<vuho_audio::AudioError>>,
    }
    impl TestAudioSource {
        fn new(keep_alive: Sender<Vec<f32>>) -> Self {
            Self {
                keep_alive,
                error: std::cell::Cell::new(None),
            }
        }

        fn with_error(keep_alive: Sender<Vec<f32>>, error: vuho_audio::AudioError) -> Self {
            Self {
                keep_alive,
                error: std::cell::Cell::new(Some(error)),
            }
        }
    }
    impl AudioSource for TestAudioSource {
        fn level_rms(&self) -> f32 {
            0.0
        }
        fn stop(self) {
            drop(self.keep_alive);
        }
        fn take_error(&self) -> Option<vuho_audio::AudioError> {
            self.error.take()
        }
    }

    fn load_models() -> Option<ParakeetModels> {
        let folder = crate::resolve_model_folder().ok()?;
        match ParakeetModels::load(&folder) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("skipping: ParakeetModels failed to load: {e}");
                None
            }
        }
    }

    fn tok(id: u32, frame: usize) -> TokenAt {
        TokenAt { id, frame }
    }

    /// Scan the real vocabulary for `count` distinct word-initial ids with
    /// plain alphanumeric text (no punctuation, no special `<...>` tokens)
    /// — real "whole word" tokens to build a deterministic merge/promotion
    /// scenario against, without hardcoding ids the shipped vocab file
    /// happens to use today.
    fn find_word_initial_ids(models: &ParakeetModels, count: usize) -> Vec<u32> {
        (0..8192u32)
            .filter(|&id| {
                models
                    .piece_info(id)
                    .is_some_and(|(is_word_initial, text)| {
                        let core = text.trim();
                        is_word_initial
                            && !core.is_empty()
                            && core.chars().all(char::is_alphanumeric)
                    })
            })
            .take(count)
            .collect()
    }

    /// D1 regression: a VAD-endpoint promotion (`promote_fresh_to_committed`)
    /// must apply the *whole* `MergeOutcome` a preceding partial computed —
    /// `keep_committed` as well as `append` — not just blindly extend
    /// `committed` with the promoted tokens.
    ///
    /// Scenario: `committed = [A, B]`. A fresh re-decode of the same open
    /// window reproduces `A, B` (within the overlap tolerance) and
    /// additionally emits `C, D` — `merge::merge`'s real contract for that
    /// case is `keep_committed: 0, append: [A, B, C, D]` (`append` INCLUDES
    /// the matched seam words, `fresh`'s copy — see `stream::merge`'s doc
    /// comment). Before this fix, promotion only ever stored/consumed the
    /// `append` half (`Accumulator::promote`, a blind `committed.extend`),
    /// so a promotion here produced `[A, B, A, B, C, D]` — this test fails
    /// on that code with exactly that duplicated sequence, and passes with
    /// `[A, B, C, D]` after the fix.
    #[test]
    fn promotion_applies_the_full_merge_outcome_not_just_append() {
        let Some(models) = load_models() else { return };

        let ids = find_word_initial_ids(&models, 4);
        assert_eq!(
            ids.len(),
            4,
            "expected at least 4 word-initial ids in the real vocab to build this scenario"
        );
        let (id_a, id_b, id_c, id_d) = (ids[0], ids[1], ids[2], ids[3]);

        let mut state = SessionState::new();
        // Seed committed = [A, B] via the same `Accumulator::apply` path a
        // real commit/promotion uses (not a private-field poke).
        state.acc.apply(
            MergeOutcome {
                keep_committed: 0,
                append: vec![tok(id_a, 10), tok(id_b, 11)],
            },
            &models,
        );
        assert_eq!(
            state.acc.committed(),
            [tok(id_a, 10), tok(id_b, 11)],
            "sanity: seeded committed = [A, B]"
        );

        // A fresh re-decode of the open window: A, B reproduced (same ids,
        // frames within `merge`'s overlap tolerance), plus newly emitted C, D.
        let emitted = vec![tok(id_a, 10), tok(id_b, 11), tok(id_c, 12), tok(id_d, 13)];
        apply_partial_merge(&mut state, emitted, |id| models.piece_info(id));
        assert_eq!(
            state.fresh_keep_committed, 0,
            "sanity: merge must recognize A, B as the matched seam and re-splice from frame 0"
        );

        let (events_tx, _events_rx) = crossbeam_channel::unbounded::<DictationEvent>();
        promote_fresh_to_committed(&mut state, &models, &events_tx);

        assert_eq!(
            state.acc.committed(),
            [tok(id_a, 10), tok(id_b, 11), tok(id_c, 12), tok(id_d, 13)],
            "promoted committed must be exactly [A, B, C, D], not a duplicated seam — got: {:?}",
            state.acc.committed()
        );
    }

    /// D3 regression: a failed commit/final-window inference must surface a
    /// recoverable `DictationEvent::Error` naming the failure (not just a
    /// log line), so a user watching the overlay sees *something* when up
    /// to one window's worth of audio silently fails to transcribe. This
    /// exercises `report_inference_failure` directly — the decision helper
    /// `infer_and_apply` calls on its `Err` arm — rather than the full
    /// `commit_window`/`finalize_tail` → real `ParakeetModels::infer_window`
    /// path: forcing a genuine `CoreML` failure deterministically would need
    /// fault injection this crate's `models: &ParakeetModels` (a concrete
    /// type, not a trait) doesn't support without a wider refactor out of
    /// scope here. The full "a real `CoreML` failure during a commit reaches
    /// the UI" path is therefore integration-BLIND; this test covers the
    /// event-construction half of that path precisely.
    #[test]
    fn report_inference_failure_sends_a_recoverable_error_naming_the_context() {
        let (events_tx, events_rx) = crossbeam_channel::unbounded::<DictationEvent>();
        let err = EngineError::CoreMl("synthetic failure for D3 regression test".to_string());

        report_inference_failure(&events_tx, "window commit", &err);

        let event = events_rx.try_recv().expect("expected an event to be sent");
        match event {
            DictationEvent::Error {
                message,
                recoverable,
                kind,
            } => {
                assert!(
                    message.contains("window commit"),
                    "message must name which inference failed, got: {message}"
                );
                assert!(
                    message.contains("synthetic failure for D3 regression test"),
                    "message must include the underlying error, got: {message}"
                );
                assert!(
                    recoverable,
                    "a single failed window must not end the session"
                );
                assert_eq!(kind, ErrorKind::Other);
            }
            other => panic!("expected DictationEvent::Error, got {other:?}"),
        }
    }

    /// D7 regression: a single chunk larger than `ADVANCE` (nothing
    /// upstream bounds chunk size to `ADVANCE`) must still get every full
    /// window it completes committed — not just the first — leaving
    /// `open_buffer` under `WINDOW_SAMPLES` afterward. Before the `if` →
    /// `while` fix, a chunk this large would commit once and then leave
    /// `open_buffer.len() >= WINDOW_SAMPLES`, which the next `infer_window`
    /// call would either violate a `debug_assert!` on (debug builds) or
    /// silently truncate the excess audio on (release builds).
    #[test]
    fn oversized_chunk_commits_every_full_window_and_leaves_a_sane_tail() {
        let Some(models) = load_models() else { return };

        let mut vad = crate::vad::Vad::new().expect("vad init must succeed");
        let mut state = SessionState::new();
        let mut cadence = PartialCadence::new(Duration::ZERO);
        let (events_tx, events_rx) = crossbeam_channel::unbounded::<DictationEvent>();

        // Two full windows' worth in one chunk — big enough that a single
        // `if`-gated commit (the pre-fix code) would leave `open_buffer`
        // still >= WINDOW_SAMPLES afterward.
        let oversized = vec![0.0f32; 2 * windower::WINDOW_SAMPLES];
        assert!(oversized.len() > windower::ADVANCE + windower::WINDOW_SAMPLES);

        handle_chunk(
            &oversized,
            &mut vad,
            &mut state,
            &models,
            &events_tx,
            true,
            &mut cadence,
        );

        assert_eq!(
            state.open_buffer.len(),
            2 * windower::WINDOW_SAMPLES - 2 * windower::ADVANCE,
            "both full windows in the oversized chunk must be committed, leaving exactly the sane tail"
        );
        assert!(
            state.open_buffer.len() < windower::WINDOW_SAMPLES,
            "post-condition every caller relies on: the open buffer is never left over-full"
        );

        let commit_partials = events_rx
            .try_iter()
            .filter(|e| matches!(e, DictationEvent::PartialTranscript { .. }))
            .count();
        assert!(
            commit_partials >= 2,
            "expected at least two commit-triggered PartialTranscript events, got {commit_partials}"
        );
    }

    /// An unexpected chunk-channel disconnect (capture thread died — no
    /// `stop` requested) must surface the *specific* `AudioError` the
    /// capture thread recorded (via `AudioSource::take_error`), not the
    /// generic "audio capture ended unexpectedly" string. This is the
    /// write-only-`take_error` fix (rule 21): before it, this test would
    /// have observed the generic message regardless of what
    /// `take_error()` returned, because nothing ever called it.
    ///
    /// `chunk_tx` is dropped with zero clones held anywhere (in particular,
    /// NOT by `audio_source` — unlike the graceful-stop `TestAudioSource`
    /// usage above, which deliberately keeps a clone alive until its own
    /// `stop()` runs) so `run_session` observes `Disconnected` on its very
    /// first `recv_timeout`, before `stop` is ever set — landing in the
    /// unexpected-disconnect branch of `handle_disconnected`.
    #[test]
    fn unexpected_disconnect_surfaces_the_recorded_audio_error() {
        let Some(models) = load_models() else { return };
        let models = crate::coreml::SendModel(models);

        let (chunk_tx, chunk_rx) = crossbeam_channel::unbounded::<Vec<f32>>();
        drop(chunk_tx); // No sends, no clones held — immediate disconnect.

        // A throwaway sender unrelated to `chunk_rx`, purely to satisfy
        // `TestAudioSource`'s field — `stop()` is never called in this
        // test, so nothing ever drops it to observe.
        let (throwaway_tx, _throwaway_rx) = crossbeam_channel::unbounded::<Vec<f32>>();
        let audio_source = TestAudioSource::with_error(
            throwaway_tx,
            vuho_audio::AudioError::StreamDied("device unplugged".to_string()),
        );

        let (events_tx, events_rx) = crossbeam_channel::unbounded::<DictationEvent>();
        let stop = Arc::new(AtomicBool::new(false));

        let session = std::thread::spawn(move || {
            let models = models;
            run_session(
                &chunk_rx,
                &events_tx,
                &stop,
                &models.0,
                audio_source,
                "en",
                Duration::ZERO,
            )
        });

        let mut error_message = None;
        for event in &events_rx {
            if let DictationEvent::Error { message, .. } = event {
                error_message = Some(message);
            }
        }
        session.join().expect("session thread panicked");

        let message = error_message.expect("expected a DictationEvent::Error");
        assert!(
            message.contains("device unplugged"),
            "expected the specific AudioError::StreamDied detail in the message, got: {message}"
        );
        assert_ne!(
            message, "audio capture ended unexpectedly",
            "must not fall back to the generic message when a specific AudioError was recorded"
        );
    }

    /// Feed `jfk.wav` in 1 s chunks through `run_session` (no `cpal`, no
    /// microphone, no wall-clock pacing — `partial_interval: Duration::ZERO`
    /// makes every chunk with speech due for a partial, so the streaming
    /// contract is exercised without an artificial `thread::sleep`) and
    /// assert: at least one `PartialTranscript` arrives before the final
    /// result, and the final text contains the expected quote. jfk.wav
    /// (~11 s) stays within a single 15 s window, so this exercises the
    /// "utterances ≤15 s must be flawless" path without touching the
    /// (now-fixed) cross-window seam behavior covered by
    /// `tests/batch_multiwindow.rs`.
    ///
    /// No wall-clock pacing between sends: `partial_interval: Duration::ZERO`
    /// makes the first speech-containing chunk immediately due for a
    /// partial, so a per-chunk sleep is not needed to make one *happen* —
    /// but setting the stop flag still has to wait for the session to have
    /// actually gotten to it (an unbounded channel means the feeder could
    /// otherwise send every chunk and flip `stop` before the session thread
    /// runs its first loop iteration at all, which would gate off cadence
    /// partials for the *entire* drained backlog — see `handle_chunk`'s
    /// `audio_live` doc comment). `saw_partial_flag`, set by the event-drain
    /// loop below the moment the first `PartialTranscript` arrives, is that
    /// wait condition — real synchronization, not a tuned sleep duration.
    #[test]
    fn streams_jfk_wav_in_chunks_and_produces_partial_then_final() {
        // 1 s @ 16 kHz — a handful of chunks is enough to exercise
        // interleaving without driving one real inference per 100 ms of
        // audio (each partial re-inference re-encodes a full 15 s padded
        // window regardless of how much of it is real audio, so the cost
        // per call is ~constant — fewer, larger chunks keeps the call count
        // and thus the test's wall time down).
        const CHUNK_SAMPLES: usize = 16_000;
        /// Upper bound on how long the feeder waits for the first partial
        /// before giving up and flipping `stop` anyway — bounds the test's
        /// worst case to "fails promptly" rather than "hangs forever" if
        /// the streaming contract really is broken.
        const SAW_PARTIAL_TIMEOUT: Duration = Duration::from_secs(30);

        let Some(samples) = crate::test_support::load_jfk_wav_f32() else {
            eprintln!("skipping: JFK_WAV/jfk.wav not found in this environment");
            return;
        };
        let Some(models) = load_models() else { return };
        // `MLModel` handles aren't `Send`; wrap for the cross-thread move,
        // same as production (`ParakeetEngine`'s `Arc<SendModel<..>>`) —
        // sound because all CoreML calls stay serialized on the one
        // session thread below.
        let models = crate::coreml::SendModel(models);

        let (chunk_tx, chunk_rx) = crossbeam_channel::unbounded::<Vec<f32>>();
        let (events_tx, events_rx) = crossbeam_channel::unbounded::<DictationEvent>();
        let stop = Arc::new(AtomicBool::new(false));
        let saw_partial_flag = Arc::new(AtomicBool::new(false));
        // The session's own `AudioSource::stop()` is the only thing that
        // may close the chunk channel (see `TestAudioSource`'s doc
        // comment) — so the feeder sends on its own clone and never
        // touches this one.
        let audio_source = TestAudioSource::new(chunk_tx.clone());

        let feed_stop = Arc::clone(&stop);
        let feed_saw_partial = Arc::clone(&saw_partial_flag);
        let feeder = std::thread::spawn(move || {
            for chunk in samples.chunks(CHUNK_SAMPLES) {
                if chunk_tx.send(chunk.to_vec()).is_err() {
                    break;
                }
            }
            // All audio sent. Wait for proof the session actually produced
            // a partial before simulating the user pressing the stop
            // hotkey — see this test's doc comment.
            let started_waiting = Instant::now();
            while !feed_saw_partial.load(Ordering::SeqCst)
                && started_waiting.elapsed() < SAW_PARTIAL_TIMEOUT
            {
                std::thread::sleep(Duration::from_millis(2));
            }
            feed_stop.store(true, Ordering::SeqCst);
        });

        let stop_for_session = Arc::clone(&stop);
        let session = std::thread::spawn(move || {
            // Rebind first: Rust 2021 disjoint closure capture would otherwise
            // capture only the `.0` field below, losing `SendModel`'s `unsafe
            // impl Send` (it applies to the whole newtype, not its field).
            let models = models;
            run_session(
                &chunk_rx,
                &events_tx,
                &stop_for_session,
                &models.0,
                audio_source,
                "en",
                Duration::ZERO,
            )
        });

        let mut saw_partial = false;
        // Drain events until the sender side (session thread) exits and
        // drops its Sender, which closes this Receiver.
        for event in &events_rx {
            if matches!(event, DictationEvent::PartialTranscript { .. }) {
                saw_partial = true;
                saw_partial_flag.store(true, Ordering::SeqCst);
            }
        }
        feeder.join().expect("feeder thread panicked");
        let result = session.join().expect("session thread panicked");
        assert!(
            saw_partial,
            "expected at least one PartialTranscript before the final result"
        );
        let lower = result.full_text.to_lowercase();
        assert!(
            lower.contains("ask not what your country can do for you"),
            "expected the JFK quote in the final transcript, got: {}",
            result.full_text
        );
    }

    #[test]
    fn activity_level_maps_silence_to_zero_and_full_scale_to_one() {
        assert!((activity_level(0.0) - 0.0).abs() < f32::EPSILON);
        assert!((activity_level(0.0) - 0.0).abs() < f32::EPSILON);
        assert!((activity_level(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn due_for_partial_requires_speech_and_elapsed_interval() {
        assert!(
            !due_for_partial(None, false, PARTIAL_INTERVAL),
            "no speech yet — never due"
        );
        assert!(
            due_for_partial(None, true, PARTIAL_INTERVAL),
            "first partial: speech and no prior timestamp"
        );
        assert!(
            !due_for_partial(Some(Instant::now()), true, PARTIAL_INTERVAL),
            "just ran — not due again until PARTIAL_INTERVAL elapses"
        );
    }

    /// Pure decision function for the VAD-endpoint flush: fires exactly once
    /// per pause (gated by the `flushed_since_speech` latch), never below the
    /// silence threshold, and never while already flushed.
    #[test]
    fn endpoint_flush_due_fires_once_at_threshold_and_not_below_it() {
        assert!(
            !endpoint_flush_due(ENDPOINT_SILENCE_MS, true),
            "already flushed since last speech — must not fire again"
        );
        assert!(
            endpoint_flush_due(ENDPOINT_SILENCE_MS, false),
            "at threshold, not yet flushed — must fire"
        );
        assert!(
            endpoint_flush_due(ENDPOINT_SILENCE_MS + 1000, false),
            "past threshold, not yet flushed — must fire"
        );
        assert!(
            !endpoint_flush_due(ENDPOINT_SILENCE_MS - 1, false),
            "below threshold — must not fire regardless of latch state"
        );
        assert!(
            !endpoint_flush_due(0, false),
            "no trailing silence at all — must not fire"
        );
    }

    /// `PartialCadence`'s flush latch: starts flushed (nothing to flush
    /// before any speech), clears on `note_speech`, and sets again on
    /// `mark_flushed` — the one-shot behaviour `endpoint_flush_due` relies
    /// on to cap the endpoint flush to once per pause.
    #[test]
    fn partial_cadence_flush_latch_transitions() {
        let mut cadence = PartialCadence::new(PARTIAL_INTERVAL);
        assert!(
            cadence.flushed_since_speech(),
            "no speech yet — nothing to flush"
        );

        cadence.note_speech();
        assert!(
            !cadence.flushed_since_speech(),
            "speech happened — a flush is now owed"
        );

        cadence.mark_flushed();
        assert!(
            cadence.flushed_since_speech(),
            "flush ran — latch set until the next speech"
        );

        // A second mark_flushed without intervening speech is idempotent.
        cadence.mark_flushed();
        assert!(cadence.flushed_since_speech());

        cadence.note_speech();
        assert!(
            !cadence.flushed_since_speech(),
            "new speech after a flush must owe another flush"
        );
    }
}
