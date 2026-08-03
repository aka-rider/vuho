//! The dictation pipeline: streaming STT → finalize → post-process → inject.
//!
//! Runs in a background thread, driven by `DictationCommand::Toggle`.
//! On start: detects language from TIS, starts the STT engine's streaming
//! session, forwards partial transcripts and activity to the UI.
//! On stop: finalizes transcription, post-processes text, and injects
//! inline (the WP6 inverted-Stop fix already returns to Idle immediately
//! on Stop via `dispatch`, so the command thread is never blocked).

use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use log::info;
use vuho_domain::{
    DictationCommand, DictationEvent, ErrorKind, InjectionOutcome, TranscriptionResult,
};
use vuho_os_integration::OsError;
use vuho_settings::SettingsStore;
use vuho_stt_engine::{EngineError, TranscriptionEngine};

use crate::Injector;

/// Time the pipeline waits for a command before polling again.
/// 100ms gives responsive hotkey handling without busy-waiting.
const COMMAND_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Pipeline state: idle or recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineState {
    Idle,
    Recording,
}

/// The dictation pipeline.
///
/// Receives commands on `command_rx`, emits events on `event_tx`.
/// Owns the STT engine and streaming receiver for the duration of a session.
pub struct DictationPipeline {
    command_rx: Receiver<DictationCommand>,
    /// Event sender (bridges pipeline thread → UI).
    event_tx: Sender<DictationEvent>,
    state: PipelineState,
    /// STT engine — app-scoped and already loaded (CONSTITUTION rule 3).
    ///
    /// Injected at construction, never built here: `TranscriptionEngine` only
    /// exists in a ready state, so starting a session costs a `start_stream`
    /// call rather than a multi-minute model load on the command thread.
    engine: Box<dyn TranscriptionEngine + Send>,
    /// Streaming event receiver from the engine.
    /// Stored per CONSTITUTION rule 1 (never drop a channel receiver).
    stream_rx: Option<Receiver<DictationEvent>>,
    /// Injected at construction (CONSTITUTION rule 5) — `handle_start` reads
    /// the configured microphone device from it on every session start.
    settings: Arc<SettingsStore>,
    /// Session language policy (ADR-009). Injected rather than called directly
    /// so tests can opt out of TIS without the pipeline having to guess whether
    /// it is under test; `None` means engine auto-detect.
    detect_language: fn() -> Option<String>,
    /// Injector: delivers post-processed text into the focused app (⌘V or
    /// clipboard fallback). Injected at construction so tests can override
    /// without touching CGEvent/clipboard APIs (CONSTITUTION rule 5). Every
    /// caller (production and every test) goes through this same field —
    /// there is no `#[cfg(test)]` seam that would let the tested path
    /// diverge from the shipped one (CONSTITUTION rule 5).
    injector: Injector,
}

/// Production language policy: match the active OS keyboard input source.
///
/// Reads the main-thread language watcher's cache — never TIS directly:
/// TIS traps (uncatchable SIGTRAP) off the main thread, and this runs on
/// the pipeline thread. `None` (watcher not installed, or unmapped
/// language) means engine auto-detect.
#[must_use]
pub(crate) fn detect_input_language() -> Option<String> {
    vuho_os_integration::cached_input_language()
}

impl DictationPipeline {
    /// Builds a pipeline around an already-loaded engine.
    ///
    /// The engine is injected (CONSTITUTION rule 5) rather than constructed
    /// here, so production and tests drive the identical path — tests pass a
    /// fake, production passes a warmed `vuho_stt_engine::ParakeetEngine`.
    ///
    /// `injector` delivers post-processed text into the focused app. In
    /// production this is `Arc::new(vuho_os_integration::inject_text)`;
    /// in tests it's a fake that records calls without touching macOS APIs.
    #[must_use]
    pub fn new(
        command_rx: Receiver<DictationCommand>,
        event_tx: Sender<DictationEvent>,
        engine: Box<dyn TranscriptionEngine + Send>,
        settings: Arc<SettingsStore>,
        detect_language: fn() -> Option<String>,
        injector: Injector,
    ) -> Self {
        Self {
            command_rx,
            event_tx,
            state: PipelineState::Idle,
            engine,
            detect_language,
            stream_rx: None,
            settings,
            injector,
        }
    }

    /// Main loop: process commands and drive the dictation session.
    ///
    /// Exits when the command channel is closed (sender dropped).
    /// The loop never breaks on Stop — the session is long-lived across
    /// many toggles.
    pub fn run(&mut self) {
        info!("pipeline: entering run loop");
        loop {
            log::trace!(
                "pipeline: polling for command (timeout={}ms)",
                COMMAND_POLL_INTERVAL.as_millis()
            );
            match self.command_rx.recv_timeout(COMMAND_POLL_INTERVAL) {
                Ok(command) => {
                    log::debug!("pipeline: received command {command:?}");
                    // Reachable states here are always Idle (Recording is
                    // entirely owned by `poll_while_recording` until it
                    // returns), so `dispatch` can only report `Continue` or
                    // `SessionStarted` — `SessionEnded` cannot occur from a
                    // command processed at this level.
                    match self.dispatch(command) {
                        Action::SessionStarted => self.poll_while_recording(),
                        Action::Continue | Action::SessionEnded => {}
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    info!("pipeline: command channel disconnected — exiting run loop");
                    break;
                }
            }
        }
    }

    /// Dispatch a single command.
    ///
    /// An explicit, exhaustive `match (command, state)` — every
    /// `(DictationCommand, PipelineState)` combination is named once, with
    /// no indirection through an "expected state" comparison. This replaced
    /// a `start_or_stop(expected)` helper whose inverted logic made `Stop`
    /// while `Recording` call `handle_start()` (comparing `self.state ==
    /// expected` where `expected` was `Recording` — the currently-recording
    /// case matched, so it "started," i.e. opened a brand-new session
    /// instead of stopping the current one) and `Stop` while `Idle` no-op
    /// only by accident of the same comparison landing on the stop branch.
    /// The regression test below (`stop_while_recording_stops_and_starts_no_new_session`)
    /// fails against the old logic and passes against this one.
    pub(crate) fn dispatch(&mut self, command: DictationCommand) -> Action {
        match (command, self.state) {
            (DictationCommand::Stop, PipelineState::Recording) => {
                info!("pipeline: Stop while Recording → stop");
                self.handle_stop();
                Action::SessionEnded
            }
            (DictationCommand::Stop, PipelineState::Idle) => {
                info!("pipeline: Stop while Idle → no-op");
                Action::Continue
            }
            // Start and Toggle behave identically while Idle (both start);
            // named together, not collapsed into "any command starts" —
            // Stop-while-Idle above and {Start,Toggle}-while-Recording below
            // are still each named on their own.
            (DictationCommand::Start | DictationCommand::Toggle, PipelineState::Idle) => {
                self.start_session()
            }
            (DictationCommand::Start, PipelineState::Recording) => {
                // A no-op: unlike the old code (finding 3), this must NOT be
                // treated as "session over" by a caller inside
                // `poll_while_recording` — that treated every `Continue` as
                // "stop polling", silently detaching the UI from a still-live
                // stream. `Action::Continue` here means exactly "state
                // unchanged, keep doing what you were doing".
                info!("pipeline: Start while Recording → no-op");
                Action::Continue
            }
            (DictationCommand::Toggle, PipelineState::Recording) => {
                info!("pipeline: Toggle while Recording → stop");
                self.handle_stop();
                Action::SessionEnded
            }
        }
    }

    /// Start a new session via `handle_start`, returning `SessionStarted`
    /// only if the engine actually started (so the caller enters the
    /// event-poll loop) — if the engine failed, `handle_start` already
    /// emitted an error and left `state` at `Idle`, so this returns
    /// `Continue` instead.
    fn start_session(&mut self) -> Action {
        self.handle_start();
        if self.state == PipelineState::Recording {
            Action::SessionStarted
        } else {
            Action::Continue
        }
    }

    /// Detect the session language, read the configured microphone, and ask
    /// the engine to start streaming — then hand the `Result` to whichever
    /// of [`Self::on_stream_started`]/[`Self::on_stream_start_failed`]
    /// applies (CONSTITUTION rule 28: prepare / call / react, split apart).
    fn handle_start(&mut self) {
        info!("pipeline: handle_start — starting session");

        // Detect language from the current OS input source.
        // None → engine auto-detect (ADR-009).
        let language = (self.detect_language)();
        info!("pipeline: language for session: {language:?}");

        let mic = self.settings.get().microphone;
        info!("pipeline: starting stream (mic={mic:?})");
        match self
            .engine
            .start_stream(language.as_deref(), mic.as_deref())
        {
            Ok(rx) => self.on_stream_started(rx),
            Err(e) => self.on_stream_start_failed(&e),
        }
    }

    /// `handle_start`'s success path: the stream is live, so it's now safe
    /// to tell the UI we're listening (CONSTITUTION rule 11 — emitting
    /// `SessionStarted` any earlier would lie about that).
    fn on_stream_started(&mut self, rx: Receiver<DictationEvent>) {
        info!("pipeline: stream started");
        self.stream_rx = Some(rx);
        self.state = PipelineState::Recording;
        self.emit(DictationEvent::SessionStarted);
    }

    /// `handle_start`'s failure path: classify the error (ADR-012 — mic
    /// denial surfaces distinctly so the overlay can prompt to grant it),
    /// emit it, and stay `Idle`.
    fn on_stream_start_failed(&mut self, err: &EngineError) {
        let (message, kind) = if matches!(err, EngineError::MicPermissionDenied) {
            log::error!("pipeline: microphone permission denied");
            (
                "Microphone permission denied".to_string(),
                ErrorKind::MicPermissionDenied,
            )
        } else {
            log::error!("pipeline: failed to start stream: {err}");
            (format!("Failed to start stream: {err}"), ErrorKind::Other)
        };
        let recoverable = !matches!(kind, ErrorKind::MicPermissionDenied);
        self.emit(DictationEvent::Error {
            message,
            recoverable,
            kind,
        });
        self.state = PipelineState::Idle;
    }

    /// Stop the active recording session: close the stream, finalize
    /// transcription (post-process + inject), return to `Idle`.
    ///
    /// Only ever called from `dispatch`'s two `(_, PipelineState::Recording)`
    /// arms — the exhaustive match there is what guarantees `self.state ==
    /// Recording` on entry, so (unlike before the WP6 dispatch rewrite) this
    /// no longer needs its own runtime "not recording, no-op" guard; a
    /// `debug_assert` keeps that invariant checked in debug/test builds
    /// without dead code in release.
    fn handle_stop(&mut self) {
        debug_assert_eq!(
            self.state,
            PipelineState::Recording,
            "handle_stop must only be called while Recording — dispatch's exhaustive match \
             is the only caller and only calls it from a Recording state"
        );
        info!("pipeline: handle_stop — stopping session");

        // Close the stream channel; any remainder is drained in poll_while_recording.
        // The engine itself outlives the session — it stays loaded for the next one.
        let _stream_rx = self.stream_rx.take();

        match self.engine.stop_stream() {
            Ok(result) => self.emit_result(result),
            Err(e) => {
                log::error!("pipeline: stream stop failed: {e}");
                self.emit(DictationEvent::Error {
                    message: format!("Stream stop failed: {e}"),
                    recoverable: true,
                    kind: ErrorKind::Other,
                });
            }
        }

        self.state = PipelineState::Idle;
    }

    /// Post-process the transcription, inject the text, and emit `SessionCompleted`.
    ///
    /// This is the "finalize" half of the stop flow (CONSTITUTION rule 28):
    /// postprocess → (blank-skip or inject) → assemble → emit, with the two
    /// injection-gate branches split into their own single-responsibility
    /// helpers below. Runs inline on the pipeline thread — the WP6
    /// inverted-Stop fix already returns to Idle immediately on Stop via
    /// `dispatch`, so the command thread is never blocked.
    fn emit_result(&mut self, result: TranscriptionResult) {
        let clean = vuho_postprocess::postprocess(&result.full_text, &result.language);

        // Don't inject blank text — it would clobber the user's clipboard.
        if vuho_domain::is_blank_transcript(&clean.text) {
            self.emit_blank_completion(result, clean.text);
            return;
        }

        let injection = self.inject_and_classify(&clean.text);
        let final_result = TranscriptionResult {
            segments: result.segments,
            full_text: clean.text,
            language: result.language,
        };
        info!(
            "pipeline: session completed ({} chars)",
            final_result.full_text.len()
        );
        self.emit(DictationEvent::SessionCompleted {
            result: final_result,
            injection,
        });
    }

    /// `emit_result`'s blank-transcript path: injection is skipped entirely
    /// (CONSTITUTION rule 11 — an honest `NothingToInject`, not a fabricated
    /// paste) and the clipboard stays untouched.
    fn emit_blank_completion(&mut self, result: TranscriptionResult, clean_text: String) {
        info!("pipeline: blank transcript — skipping injection, clipboard untouched");
        let final_result = TranscriptionResult {
            segments: result.segments,
            full_text: clean_text,
            language: result.language,
        };
        self.emit(DictationEvent::SessionCompleted {
            result: final_result,
            injection: InjectionOutcome::NothingToInject,
        });
    }

    /// `emit_result`'s non-blank path: deliver `text` to the focused app via
    /// the constructor-injected [`Injector`] and classify the outcome.
    /// Every caller — production and every test — goes through this same
    /// `self.injector` call; there is no compile-time test seam.
    fn inject_and_classify(&self, text: &str) -> InjectionOutcome {
        let injection = injection_outcome((self.injector)(text));
        if !matches!(injection, InjectionOutcome::Inserted) {
            log::warn!("session injection degraded: {injection:?}");
        }
        injection
    }

    /// Poll the streaming engine and command channel while recording.
    ///
    /// Forwards engine events to both internal and external senders.
    /// Returns once the session ends (`Stop`/`Toggle` while `Recording`) or
    /// the command channel disconnects — a no-op command (e.g. `Start`
    /// while already `Recording`) keeps this loop running instead of
    /// exiting it (finding 3).
    fn poll_while_recording(&mut self) {
        while let Some(stream_rx) = &self.stream_rx {
            crossbeam_channel::select! {
                recv(stream_rx) -> event => {
                    if self.handle_stream_event_while_recording(event) {
                        return;
                    }
                }
                recv(self.command_rx) -> cmd => {
                    if self.handle_command_while_recording(cmd) {
                        return;
                    }
                }
            }
        }
    }

    /// Handle one event from the streaming engine while recording. Returns
    /// `true` if `poll_while_recording` should stop polling (the stream
    /// ended, whether by error or by disconnect).
    fn handle_stream_event_while_recording(
        &mut self,
        event: Result<DictationEvent, crossbeam_channel::RecvError>,
    ) -> bool {
        match event {
            // A fatal engine error (`recoverable: false`) means the stream
            // is dead. Just forwarding it would show the error while
            // leaving the state machine in Recording, so the next toggle
            // would read as "stop" and the session would be stuck — abort.
            Ok(
                event @ DictationEvent::Error {
                    recoverable: false, ..
                },
            ) => {
                log::error!("pipeline: stream died mid-session — aborting");
                self.emit(event);
                self.abort_session();
                true
            }
            // A recoverable error (e.g. one failed partial/window
            // inference) is diagnostic, not fatal: the stream is still
            // alive and later windows can still succeed, so the session
            // must keep polling for further partials rather than being
            // torn down over a single blip.
            Ok(event) => {
                self.emit(event);
                false
            }
            // Sender gone: nothing more will ever arrive (CONSTITUTION rule 10).
            Err(_) => {
                log::warn!("pipeline: stream channel disconnected");
                self.abort_session();
                true
            }
        }
    }

    /// Handle one command received while recording. Returns `true` if
    /// `poll_while_recording` should stop polling.
    fn handle_command_while_recording(
        &mut self,
        cmd: Result<DictationCommand, crossbeam_channel::RecvError>,
    ) -> bool {
        let Ok(cmd) = cmd else {
            // Disconnected: the command sender was dropped (e.g. the
            // owning `DictationSession` was torn down) mid-recording.
            // Ignoring this would busy-spin `select!` on an
            // ever-ready-to-error channel with a live microphone still open
            // (CONSTITUTION rule 10) — abort the session and stop polling
            // instead.
            log::warn!("pipeline: command channel disconnected mid-session — aborting");
            self.abort_session();
            return true;
        };
        match self.dispatch(cmd) {
            // The session ended (Stop/Toggle while Recording): stop polling.
            Action::SessionEnded => true,
            // A no-op (e.g. Start while already Recording): keep polling —
            // partials must keep flowing.
            Action::Continue => false,
            // Unreachable: `dispatch` only returns `SessionStarted` from an
            // Idle state, and this loop only runs while `self.state ==
            // Recording`.
            Action::SessionStarted => {
                debug_assert!(
                    false,
                    "dispatch cannot start a new session while already Recording"
                );
                false
            }
        }
    }

    /// Tear down a session that failed rather than finished.
    ///
    /// No `SessionCompleted` and no injection: the error was already emitted,
    /// and there is no transcript worth pasting. The stream handle stays valid
    /// after an error by FFI contract, so it must still be released.
    fn abort_session(&mut self) {
        self.stream_rx.take();
        if let Err(e) = self.engine.stop_stream() {
            log::warn!("pipeline: stop_stream during abort failed: {e}");
        }
        self.state = PipelineState::Idle;
    }

    fn emit(&self, event: DictationEvent) {
        // Unbounded channels never block, so send failures only happen
        // when all receivers are dropped (i.e. the consumer has exited).
        // Log a warning so this isn't a silent failure in production.
        if self.event_tx.send(event).is_err() {
            log::warn!("event channel closed — UI consumer gone");
        }
    }
}

/// Map the result of injecting the final text into `InjectionOutcome`.
///
/// A completed session is always a *transcription* success, even when the
/// paste keystroke degrades to clipboard-only — so this never produces an
/// `Error` event, only the `injection` payload of `SessionCompleted`.
pub(crate) fn injection_outcome(result: Result<(), OsError>) -> InjectionOutcome {
    match result {
        Ok(()) => InjectionOutcome::Inserted,
        Err(OsError::SecureInputActive) => InjectionOutcome::ClipboardOnly {
            reason: "Secure input active — copied to clipboard, paste manually".into(),
        },
        Err(OsError::InjectionFailed) => InjectionOutcome::ClipboardOnly {
            reason: "Paste keystroke failed — copied to clipboard, paste manually".into(),
        },
        Err(e) => InjectionOutcome::Failed {
            reason: e.to_string(),
        },
    }
}

/// Result of dispatching a command — three honest, distinct outcomes
/// (finding 3: the old two-variant `Continue`/`PollRecording` shape made
/// `poll_while_recording` treat *any* `Continue` as "session over", so a
/// no-op like `Start` while already `Recording` silently exited the poll
/// loop and detached the UI from a still-live stream).
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    /// The command was a no-op: state is unchanged, and the caller should
    /// keep doing whatever it was already doing (stay in the outer `run`
    /// loop, or keep polling inside `poll_while_recording`).
    Continue,
    /// A new recording session just started (an Idle → Recording
    /// transition happened during this call) — the caller should enter
    /// `poll_while_recording`.
    SessionStarted,
    /// The active recording session just ended (a Recording → Idle
    /// transition happened during this call, via `handle_stop`) —
    /// `poll_while_recording` should stop polling and return.
    SessionEnded,
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use super::*;
    use crate::test_support::fake_injector;

    /// Language policy for tests: never touch TIS. Calling macOS APIs from a
    /// test thread crashes with SIGTRAP rather than failing cleanly.
    fn no_language() -> Option<String> {
        None
    }

    /// A fresh `SettingsStore` backed by a per-test temp path — never
    /// touches the real `~/.config/vuho/settings.json`.
    fn test_settings() -> Arc<SettingsStore> {
        Arc::new(SettingsStore::new_temp("pipeline"))
    }

    // ── Fake engine for unit testing ───────────────────────────────────────

    /// Records the `input_device` passed to `start_stream`, for tests that
    /// verify the pipeline threads the configured microphone through.
    /// `None` means `start_stream` was never called; `Some(None)` means it
    /// was called with no device configured.
    type DeviceProbe = Arc<Mutex<Option<Option<String>>>>;

    /// The sender side of every stream a `FakeEngine` has started — the test's
    /// stand-in for the real `TranscriptionEngine` (`ParakeetEngine` in
    /// production) pushing events at us.
    type StreamTaps = Arc<Mutex<Vec<Sender<DictationEvent>>>>;

    /// A no-op transcription engine for unit tests.
    ///
    /// `start_stream` returns a live-but-quiet receiver (no partial
    /// transcripts) and records the `input_device` it was called with, if a
    /// probe was supplied. `stop_stream` returns a configurable
    /// `TranscriptionResult`.
    struct FakeEngine {
        stop_result: TranscriptionResult,
        device_probe: Option<DeviceProbe>,
        /// Holds each started stream's sender so the channel stays open for
        /// the session, exactly as the real engine's `StreamContext` does. A
        /// dropped sender means "this stream is dead" — which the pipeline
        /// acts on — so a fake that dropped it would be simulating a failure.
        /// Shared, so a test can push events as the engine would.
        stream_txs: StreamTaps,
        /// Number of `start_stream` calls so far — `stream_txs.lock().len()`
        /// already tracks this, but a dedicated counter reads clearer at
        /// call sites that just want "how many times did this fire," not
        /// the taps themselves.
        start_calls: Arc<std::sync::atomic::AtomicUsize>,
        /// Number of `stop_stream` calls so far.
        stop_calls: Arc<std::sync::atomic::AtomicUsize>,
        /// If `true`, the *next* `start_stream` call fails with
        /// `EngineError::LoadFailed` and clears this flag — a stand-in for a
        /// dropped/discarded `Start` command (e.g. arriving during model
        /// warmup, `wiring.rs`), which leaves the pipeline `Idle` even though
        /// the caller believes it sent `Start`.
        fail_next_start: Arc<std::sync::atomic::AtomicBool>,
    }

    impl FakeEngine {
        fn new(final_text: &str, language: &str) -> Self {
            Self {
                stop_result: TranscriptionResult {
                    segments: vec![],
                    full_text: final_text.to_string(),
                    language: language.to_string(),
                },
                device_probe: None,
                stream_txs: StreamTaps::default(),
                start_calls: Arc::default(),
                stop_calls: Arc::default(),
                fail_next_start: Arc::default(),
            }
        }

        /// Like `new`, but records every `start_stream` device argument into
        /// `probe` (shared with the test so it can assert after `dispatch`).
        fn with_device_probe(final_text: &str, language: &str, probe: DeviceProbe) -> Self {
            Self {
                device_probe: Some(probe),
                ..Self::new(final_text, language)
            }
        }
    }

    impl TranscriptionEngine for FakeEngine {
        fn transcribe(
            &self,
            _samples: &[f32],
            _language: Option<&str>,
        ) -> Result<TranscriptionResult, vuho_stt_engine::EngineError> {
            Ok(self.stop_result.clone())
        }
        fn unload(&self) {}
        fn start_stream(
            &self,
            _language: Option<&str>,
            input_device: Option<&str>,
        ) -> Result<Receiver<DictationEvent>, vuho_stt_engine::EngineError> {
            self.start_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self
                .fail_next_start
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                return Err(vuho_stt_engine::EngineError::LoadFailed(
                    "simulated dropped/discarded start".to_string(),
                ));
            }
            if let Some(probe) = &self.device_probe {
                *probe.lock().unwrap() = Some(input_device.map(str::to_string));
            }
            // Quiet, but live: no partial transcripts, sender kept alive.
            let (tx, rx) = crossbeam_channel::unbounded();
            self.stream_txs.lock().unwrap().push(tx);
            Ok(rx)
        }
        fn stop_stream(&self) -> Result<TranscriptionResult, vuho_stt_engine::EngineError> {
            self.stop_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(self.stop_result.clone())
        }
    }

    /// A stream that dies mid-session (`recoverable: false`) must return the
    /// pipeline to `Idle`.
    ///
    /// The engine reports such deaths out-of-band as an `Error` on the stream
    /// channel. Merely forwarding it would leave the state machine in
    /// `Recording`, so the next hotkey press would be read as "stop" and the
    /// user could never start again.
    #[test]
    fn stream_error_aborts_the_session_and_returns_to_idle() {
        let (_command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();

        let fake_engine = FakeEngine::new("unused", "en");
        let taps = Arc::clone(&fake_engine.stream_txs);
        let (injector, _received) = fake_injector();
        let mut pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            test_settings(),
            no_language,
            injector,
        );

        assert!(matches!(
            pipeline.dispatch(DictationCommand::Toggle),
            Action::SessionStarted
        ));
        assert_eq!(pipeline.state, PipelineState::Recording);

        // Simulate the engine reporting a dead stream, as `on_stream_error`
        // does in production.
        taps.lock()
            .unwrap()
            .first()
            .expect("start_stream must have opened a stream")
            .send(DictationEvent::Error {
                message: "audio engine died".to_string(),
                recoverable: false,
                kind: ErrorKind::Other,
            })
            .unwrap();
        pipeline.poll_while_recording();

        assert_eq!(pipeline.state, PipelineState::Idle);

        let mut saw_error = false;
        while let Ok(ev) = event_rx.try_recv() {
            match ev {
                DictationEvent::Error { .. } => saw_error = true,
                DictationEvent::SessionCompleted { .. } => {
                    panic!("a failed session must not report completion")
                }
                _ => {}
            }
        }
        assert!(saw_error, "the error must reach the UI");
    }

    /// A *recoverable* engine error (e.g. a single failed partial/window
    /// inference — `recoverable: true`) must reach the UI but NOT end the
    /// session: the stream is still alive, and later windows can still
    /// succeed, so `poll_while_recording` must keep polling for further
    /// partials rather than treating one blip as stream death.
    #[test]
    fn recoverable_stream_error_reaches_ui_but_does_not_end_the_session() {
        let (_command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();

        let fake_engine = FakeEngine::new("unused", "en");
        let taps = Arc::clone(&fake_engine.stream_txs);
        let (injector, _received) = fake_injector();
        let mut pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            test_settings(),
            no_language,
            injector,
        );

        assert!(matches!(
            pipeline.dispatch(DictationCommand::Toggle),
            Action::SessionStarted
        ));
        assert_eq!(pipeline.state, PipelineState::Recording);

        let tap = taps
            .lock()
            .unwrap()
            .first()
            .expect("start_stream must have opened a stream")
            .clone();
        tap.send(DictationEvent::Error {
            message: "partial inference failed".to_string(),
            recoverable: true,
            kind: ErrorKind::Other,
        })
        .unwrap();

        // Receive straight from the stream tap (the same channel
        // `poll_while_recording`'s `select!` would receive from) and drive
        // `handle_stream_event_while_recording` directly with it — mirrors
        // this file's other white-box tests, which call pipeline internals
        // rather than spawning a thread.
        let stream_event = pipeline
            .stream_rx
            .as_ref()
            .expect("session must have a live stream_rx")
            .recv()
            .expect("the error just sent on the tap must be receivable");
        let handled = pipeline.handle_stream_event_while_recording(Ok(stream_event));
        assert!(
            !handled,
            "a recoverable error must not signal poll_while_recording to stop"
        );
        assert_eq!(
            pipeline.state,
            PipelineState::Recording,
            "a recoverable error must not abort the session"
        );

        let mut saw_error = false;
        while let Ok(ev) = event_rx.try_recv() {
            if matches!(ev, DictationEvent::Error { .. }) {
                saw_error = true;
            }
        }
        assert!(saw_error, "the recoverable error must still reach the UI");

        // The session is still recording and the tap is still open — stop
        // it cleanly so the test doesn't leak a live stream_rx.
        assert!(matches!(
            pipeline.dispatch(DictationCommand::Toggle),
            Action::SessionEnded
        ));
    }

    /// Verify that the pipeline runs post-process + inject inline on the
    /// command thread: stopping hands the transcript to `emit_result` which
    /// calls `vuho_postprocess::postprocess`, then injects, and emits
    /// `SessionCompleted` directly — no intermediate `Processing` event.
    ///
    /// Uses a fake `TranscriptionEngine` (no real STT) and a fake injector
    /// (no real macOS APIs) so this only exercises the pipeline's own wiring
    /// — the post-processed text is verified by checking the injector
    /// received cleaned text (fillers removed).
    #[test]
    fn pipeline_postprocesses_and_injects_inline_on_stop() {
        let (_command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let (injector, received) = fake_injector();

        // "um Hello world." → postprocess removes "um" → "Hello world."
        let fake_engine = FakeEngine::new("um Hello world.", "en");
        let mut pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            test_settings(),
            no_language,
            injector,
        );

        // Drive the pipeline manually: dispatch commands and poll events.
        // This avoids spawning a thread where macOS API crashes would
        // produce SIGTRAP instead of a proper panic message.

        // Idle → start
        let action = pipeline.dispatch(DictationCommand::Toggle);
        assert!(matches!(action, Action::SessionStarted));

        // Verify SessionStarted
        let started = event_rx.recv_timeout(Duration::from_millis(500)).unwrap();
        assert!(matches!(started, DictationEvent::SessionStarted));

        // Drain any PartialTranscript/Activity events from the fake engine
        while let Ok(ev) = event_rx.try_recv() {
            if matches!(ev, DictationEvent::SessionCompleted { .. }) {
                panic!("SessionCompleted should not arrive before Stop");
            }
        }

        // Recording → stop: the command thread must return to Idle right
        // away. Post-process + inject runs inline in emit_result.
        let action = pipeline.dispatch(DictationCommand::Toggle);
        assert!(matches!(action, Action::SessionEnded));
        assert_eq!(pipeline.state, PipelineState::Idle);

        // The pipeline emits SessionCompleted directly — no Processing event.
        let completed = event_rx.recv_timeout(Duration::from_millis(500)).unwrap();
        assert!(
            matches!(completed, DictationEvent::SessionCompleted { .. }),
            "Expected SessionCompleted, got {completed:?}"
        );

        if let DictationEvent::SessionCompleted { result, .. } = completed {
            // Postprocess removes "um" filler and normalizes spacing.
            assert!(
                !result.full_text.to_lowercase().contains("um"),
                "postprocess should remove 'um' filler, got: {}",
                result.full_text
            );
            assert_eq!(result.language, "en");
            assert_eq!(result.segments.len(), 0);
        }

        // The injector received the post-processed text.
        let received = received.lock().unwrap();
        assert_eq!(received.len(), 1);
        assert!(
            !received[0].to_lowercase().contains("um"),
            "injector should receive post-processed text, got: {}",
            received[0]
        );
    }

    /// Verify that the full `run()` loop processes commands from the channel,
    /// starts and stops a session, and emits the expected events end-to-end.
    ///
    /// Exercises `poll_while_recording` via the real `run()` thread, ensuring
    /// the `crossbeam_channel::select!` loop exits cleanly when a Stop command
    /// arrives on the command channel.
    #[test]
    fn run_loop_full_session_lifecycle() {
        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let (injector, _received) = fake_injector();

        let fake_engine = FakeEngine::new("Hello world.", "en");
        let pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            test_settings(),
            no_language,
            injector,
        );

        // Run the pipeline in a background thread (simulates production usage).
        // The run() loop must NOT exit on a command-channel timeout — the real
        // pipeline thread has to stay alive across many toggles over the life
        // of the process (a hotkey press an hour from now must still reach
        // it). It exits only when every `command_tx` clone is dropped, which
        // this test does explicitly below before joining.
        let handle = std::thread::spawn(move || {
            let mut pipeline = pipeline;
            pipeline.run();
        });

        // Start the session via Toggle. The command channel is unbounded, so
        // this send lands whenever `run()`'s loop reaches its first `recv` —
        // no sleep needed to "wait for the loop to start" (CONSTITUTION
        // rule 32); `recv_timeout` below is the actual synchronization point.
        command_tx.send(DictationCommand::Toggle).unwrap();

        // Verify SessionStarted.
        let started = event_rx.recv_timeout(Duration::from_millis(500)).unwrap();
        assert!(matches!(started, DictationEvent::SessionStarted));

        // Stop the session via Toggle (delivered to poll_while_recording's select!).
        command_tx.send(DictationCommand::Toggle).unwrap();

        // Verify SessionCompleted arrives directly — inline post-process + inject,
        // no intermediate Processing event.
        let completed = event_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            matches!(completed, DictationEvent::SessionCompleted { .. }),
            "Expected SessionCompleted, got {completed:?}"
        );

        // Drop the last `command_tx` so `run()`'s `recv_timeout` observes
        // `Disconnected` (within one `COMMAND_POLL_INTERVAL`) and exits —
        // otherwise `join()` below would block forever.
        drop(command_tx);
        handle.join().expect("pipeline thread panicked");
    }

    /// The pipeline must read the configured microphone device from
    /// `settings` and pass it through to `engine.start_stream` unchanged.
    #[test]
    fn pipeline_passes_configured_microphone_to_engine() {
        let (_command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, _event_rx) = crossbeam_channel::unbounded();

        let settings = test_settings();
        settings
            .update(|s| s.microphone = Some("Test Mic".to_string()))
            .unwrap();

        let probe: DeviceProbe = Arc::new(Mutex::new(None));
        let fake_engine = FakeEngine::with_device_probe("Hello world.", "en", Arc::clone(&probe));
        let (injector, _received) = fake_injector();
        let mut pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            settings,
            no_language,
            injector,
        );

        // Idle → start: should call start_stream with the configured device.
        let action = pipeline.dispatch(DictationCommand::Toggle);
        assert!(matches!(action, Action::SessionStarted));

        let recorded = probe.lock().unwrap().clone();
        assert_eq!(recorded, Some(Some("Test Mic".to_string())));
    }

    /// Regression test for the inverted-Stop bug (WP6, root cause: the old
    /// `start_or_stop(expected)` compared `self.state == expected`, and
    /// `Stop` passed `expected = Recording` — so Stop-while-Recording
    /// matched the "state equals expected" branch and called
    /// `handle_start()`, opening a brand-new session, instead of stopping
    /// the current one). `dispatch`'s new exhaustive `match (command,
    /// state)` makes this impossible by construction: `(Stop, Recording)`
    /// is its own named arm that only ever calls `handle_stop`.
    #[test]
    fn stop_while_recording_stops_and_starts_no_new_session() {
        use std::sync::atomic::Ordering;

        let (_command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, _event_rx) = crossbeam_channel::unbounded();

        let fake_engine = FakeEngine::new("Hello world.", "en");
        let start_calls = Arc::clone(&fake_engine.start_calls);
        let stop_calls = Arc::clone(&fake_engine.stop_calls);
        let (injector, _received) = fake_injector();
        let mut pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            test_settings(),
            no_language,
            injector,
        );

        // Idle → Start.
        assert!(matches!(
            pipeline.dispatch(DictationCommand::Start),
            Action::SessionStarted
        ));
        assert_eq!(pipeline.state, PipelineState::Recording);
        assert_eq!(start_calls.load(Ordering::SeqCst), 1);
        assert_eq!(stop_calls.load(Ordering::SeqCst), 0);

        // Recording → Stop: must stop, and must NOT call start_stream again.
        let action = pipeline.dispatch(DictationCommand::Stop);
        assert!(matches!(action, Action::SessionEnded));
        assert_eq!(
            pipeline.state,
            PipelineState::Idle,
            "Stop while Recording must return the pipeline to Idle"
        );
        assert_eq!(
            start_calls.load(Ordering::SeqCst),
            1,
            "Stop while Recording must NOT call start_stream again — that was the inverted-Stop bug"
        );
        assert_eq!(stop_calls.load(Ordering::SeqCst), 1);

        // Stop while already Idle must be a no-op: no crash, no restart.
        let action = pipeline.dispatch(DictationCommand::Stop);
        assert!(matches!(action, Action::Continue));
        assert_eq!(pipeline.state, PipelineState::Idle);
        assert_eq!(start_calls.load(Ordering::SeqCst), 1);
        assert_eq!(stop_calls.load(Ordering::SeqCst), 1);
    }

    /// Regression test for the level-triggered `CapsLock` fix: under the old
    /// edge-triggered `Toggle` scheme, a single dropped/discarded command
    /// (e.g. one arriving during model warmup, see `wiring.rs`) inverted the
    /// hotkey mapping for the rest of the run. Level-triggering sends
    /// `Start`/`Stop` from the LED's own state (`caps_lock_command` in
    /// `vuho-os-integration`), and `dispatch` already treats `(Start,
    /// Recording)` and `(Stop, Idle)` as no-ops — so a dropped `Start` must
    /// leave the *next* `Stop` a harmless no-op and the *next* `Start` after
    /// that must still open a session, rather than the old scheme's
    /// permanent inversion.
    #[test]
    fn dropped_start_desyncs_but_next_taps_self_heal() {
        use std::sync::atomic::Ordering;

        let (_command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, _event_rx) = crossbeam_channel::unbounded();

        let fake_engine = FakeEngine::new("Hello world.", "en");
        let start_calls = Arc::clone(&fake_engine.start_calls);
        fake_engine.fail_next_start.store(true, Ordering::SeqCst);
        let (injector, _received) = fake_injector();
        let mut pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            test_settings(),
            no_language,
            injector,
        );

        // LED on → Start, but the engine fails to start (simulating a
        // dropped/discarded command elsewhere): the pipeline stays Idle.
        let action = pipeline.dispatch(DictationCommand::Start);
        assert!(matches!(action, Action::Continue));
        assert_eq!(pipeline.state, PipelineState::Idle);
        assert_eq!(start_calls.load(Ordering::SeqCst), 1);

        // LED off → Stop: must be a harmless no-op, NOT a session start (the
        // inversion the old Toggle scheme would have produced here).
        let action = pipeline.dispatch(DictationCommand::Stop);
        assert!(matches!(action, Action::Continue));
        assert_eq!(pipeline.state, PipelineState::Idle);
        assert_eq!(start_calls.load(Ordering::SeqCst), 1);

        // LED on again → Start: must actually start a session now that the
        // engine succeeds.
        let action = pipeline.dispatch(DictationCommand::Start);
        assert!(matches!(action, Action::SessionStarted));
        assert_eq!(pipeline.state, PipelineState::Recording);
        assert_eq!(start_calls.load(Ordering::SeqCst), 2);
    }

    /// Verify the constructor-supplied injector receives the post-processed
    /// text (not the raw engine text). The injector is passed through the
    /// pipeline → `emit_result`, which calls `vuho_postprocess::postprocess`
    /// then `(self.injector)(&clean.text)`.
    #[test]
    fn pipeline_injects_via_the_constructor_supplied_injector() {
        let (_command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let (injector, received) = fake_injector();

        let fake_engine = FakeEngine::new("Hello, constructor-injected world.", "en");
        let mut pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            test_settings(),
            no_language,
            injector,
        );

        assert!(matches!(
            pipeline.dispatch(DictationCommand::Toggle),
            Action::SessionStarted
        ));
        assert!(matches!(
            pipeline.dispatch(DictationCommand::Toggle),
            Action::SessionEnded
        ));

        // Wait for the pipeline's SessionCompleted (proof it finished
        // the post-process + inject sequence inline).
        let mut got_completed = false;
        while let Ok(event) = event_rx.recv_timeout(Duration::from_secs(2)) {
            if matches!(event, DictationEvent::SessionCompleted { .. }) {
                got_completed = true;
                break;
            }
        }
        assert!(got_completed, "expected SessionCompleted from the pipeline");

        let received = received.lock().unwrap();
        assert_eq!(
            received.as_slice(),
            &["Hello, constructor-injected world.".to_string()],
            "the constructor-supplied Injector must receive the post-processed text"
        );
    }

    /// A silent/empty session must not touch the user's clipboard: the
    /// pipeline's injection gate skips the injector entirely for a
    /// blank transcript and reports `NothingToInject` — an honest
    /// `SessionCompleted` (CONSTITUTION rule 11) instead of a fabricated
    /// paste. Drives the real `emit_result` path with a `FakeEngine`
    /// yielding an empty transcript and the recording `fake_injector`.
    #[test]
    fn blank_transcript_session_skips_injection_and_reports_nothing_to_inject() {
        let (_command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let (injector, received) = fake_injector();

        let fake_engine = FakeEngine::new("", "en");
        let mut pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            test_settings(),
            no_language,
            injector,
        );

        assert!(matches!(
            pipeline.dispatch(DictationCommand::Toggle),
            Action::SessionStarted
        ));
        assert!(matches!(
            pipeline.dispatch(DictationCommand::Toggle),
            Action::SessionEnded
        ));

        // Wait for the pipeline's SessionCompleted and check its
        // injection outcome tells the truth: nothing was injected.
        let mut completed_injection = None;
        while let Ok(event) = event_rx.recv_timeout(Duration::from_secs(2)) {
            if let DictationEvent::SessionCompleted { injection, .. } = event {
                completed_injection = Some(injection);
                break;
            }
        }
        assert_eq!(
            completed_injection,
            Some(InjectionOutcome::NothingToInject),
            "a blank-transcript session must report NothingToInject"
        );
        // The injector must never have been called — the clipboard stays
        // untouched.
        assert!(
            received.lock().unwrap().is_empty(),
            "the injector must not be called for a blank transcript"
        );
    }

    /// Property-style: drive the pipeline through many seeded-random
    /// `Toggle`/`Start`/`Stop` sequences and assert the state machine never
    /// double-starts (matching `start_calls`/`stop_calls` invariant at
    /// every step) and never wedges (every dispatch returns promptly — no
    /// wall-clock waits anywhere in this test, `FakeEngine` never blocks).
    #[test]
    fn dispatch_never_double_starts_or_wedges_under_random_command_sequences() {
        use std::sync::atomic::Ordering;

        /// A tiny deterministic PRNG (xorshift32) — no crate dependency
        /// needed for a handful of 3-way coin flips per test iteration.
        struct Xorshift32(u32);
        impl Xorshift32 {
            fn next_command(&mut self) -> DictationCommand {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 17;
                self.0 ^= self.0 << 5;
                match self.0 % 3 {
                    0 => DictationCommand::Start,
                    1 => DictationCommand::Stop,
                    _ => DictationCommand::Toggle,
                }
            }
        }

        for seed in [1_u32, 42, 12345, 999_999, 0xDEAD_BEEF] {
            let (_command_tx, command_rx) = crossbeam_channel::unbounded();
            let (event_tx, _event_rx) = crossbeam_channel::unbounded();

            let fake_engine = FakeEngine::new("irrelevant", "en");
            let start_calls = Arc::clone(&fake_engine.start_calls);
            let stop_calls = Arc::clone(&fake_engine.stop_calls);
            let (injector, _received) = fake_injector();
            let mut pipeline = DictationPipeline::new(
                command_rx,
                event_tx,
                Box::new(fake_engine),
                test_settings(),
                no_language,
                injector,
            );

            let mut rng = Xorshift32(seed);
            for step in 0..200 {
                let _ = pipeline.dispatch(rng.next_command());

                // Invariant, checked after every single command: the
                // pipeline's own state must always agree with the call
                // counters — Recording means exactly one more start than
                // stop, Idle means they're equal. A "double start" (the
                // inverted-Stop bug's signature) would show up here as
                // start_calls running ahead by 2+ while still Recording,
                // or the counters going out of sync entirely.
                let starts = start_calls.load(Ordering::SeqCst);
                let stops = stop_calls.load(Ordering::SeqCst);
                match pipeline.state {
                    PipelineState::Recording => assert_eq!(
                        starts,
                        stops + 1,
                        "seed={seed} step={step}: Recording but starts({starts}) != stops({stops}) + 1"
                    ),
                    PipelineState::Idle => assert_eq!(
                        starts, stops,
                        "seed={seed} step={step}: Idle but starts({starts}) != stops({stops})"
                    ),
                }
            }
        }
    }

    // ── Finding 2: a disconnected command channel must abort, not spin ─────

    /// If the command sender is dropped mid-recording (e.g. the owning
    /// `DictationSession` is torn down while a session is live),
    /// `poll_while_recording` must observe `Disconnected`, abort the
    /// session (stopping the still-live mic stream), and return — not
    /// busy-spin forever re-selecting on an already-closed channel
    /// (CONSTITUTION rule 10).
    ///
    /// Runs `poll_while_recording` on a background thread and bounds the
    /// wait with `recv_timeout` — a safety net against the pre-fix
    /// behavior hanging the test suite, not a pacing sleep (rule 32): the
    /// fixed code returns promptly, well under the bound.
    #[test]
    fn command_channel_disconnect_mid_recording_aborts_without_spinning() {
        use std::sync::atomic::Ordering;

        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, _event_rx) = crossbeam_channel::unbounded();

        let fake_engine = FakeEngine::new("Hello world.", "en");
        let stop_calls = Arc::clone(&fake_engine.stop_calls);
        let (injector, _received) = fake_injector();
        let mut pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            test_settings(),
            no_language,
            injector,
        );

        assert!(matches!(
            pipeline.dispatch(DictationCommand::Toggle),
            Action::SessionStarted
        ));
        assert_eq!(pipeline.state, PipelineState::Recording);

        // Every command_tx clone gone → the channel disconnects.
        drop(command_tx);

        let (done_tx, done_rx) = crossbeam_channel::bounded(1);
        let handle = std::thread::spawn(move || {
            pipeline.poll_while_recording();
            let _ = done_tx.send(pipeline);
        });
        let pipeline = done_rx.recv_timeout(Duration::from_secs(5)).expect(
            "poll_while_recording must return promptly on a disconnected \
                 command channel, not busy-spin",
        );
        handle.join().expect("poll_while_recording thread panicked");

        assert_eq!(
            pipeline.state,
            PipelineState::Idle,
            "a disconnected command channel must abort the session"
        );
        assert_eq!(
            stop_calls.load(Ordering::SeqCst),
            1,
            "abort_session must stop the still-live engine stream"
        );
    }

    // ── Finding 3: a no-op command must not exit the poll loop ─────────────

    /// `Start` while already `Recording` is a no-op (`dispatch`'s
    /// `(Start, Recording)` arm) — `poll_while_recording` must keep polling
    /// afterward, not treat the no-op as "session over" (the old
    /// two-variant `Action` shape's bug: `poll_while_recording` returned on
    /// *any* `Continue`, silently detaching the UI from a still-live
    /// stream). `Stop` afterward must still end the session normally.
    #[test]
    fn start_while_recording_is_a_noop_that_keeps_polling_for_partials() {
        use std::sync::atomic::Ordering;

        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let fake_engine = FakeEngine::new("Hello world.", "en");
        let taps = Arc::clone(&fake_engine.stream_txs);
        let start_calls = Arc::clone(&fake_engine.start_calls);
        let (injector, _received) = fake_injector();
        let mut pipeline = DictationPipeline::new(
            command_rx,
            event_tx,
            Box::new(fake_engine),
            test_settings(),
            no_language,
            injector,
        );

        assert!(matches!(
            pipeline.dispatch(DictationCommand::Toggle),
            Action::SessionStarted
        ));
        assert!(matches!(
            event_rx.recv_timeout(Duration::from_millis(500)),
            Ok(DictationEvent::SessionStarted)
        ));

        // poll_while_recording blocks on select!, so drive it on a
        // background thread; the pipeline is handed back through the
        // thread's return value once it exits.
        let handle = std::thread::spawn(move || {
            pipeline.poll_while_recording();
            pipeline
        });

        // A Start while already Recording must be a no-op that does NOT
        // exit the select! loop.
        command_tx.send(DictationCommand::Start).unwrap();

        // Prove the loop is still alive: a partial transcript sent
        // afterward must still reach the UI. Bounded by recv_timeout, not
        // a sleep — if the loop had wrongly exited on the no-op, this call
        // times out instead of hanging.
        let tap = taps
            .lock()
            .unwrap()
            .first()
            .expect("start_stream must have opened a stream")
            .clone();
        tap.send(DictationEvent::PartialTranscript {
            confirmed_text: "hi".to_string(),
            unconfirmed_text: String::new(),
        })
        .unwrap();
        let ev = event_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("poll_while_recording must still forward events after a no-op Start");
        assert!(matches!(ev, DictationEvent::PartialTranscript { .. }));

        assert_eq!(
            start_calls.load(Ordering::SeqCst),
            1,
            "Start while Recording must not restart the stream"
        );

        // Stop must still end the session and return the poll loop.
        command_tx.send(DictationCommand::Stop).unwrap();
        let pipeline = handle.join().expect("poll_while_recording thread panicked");
        assert_eq!(pipeline.state, PipelineState::Idle);
    }
}
