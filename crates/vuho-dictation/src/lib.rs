//! Dictation session orchestration.
//!
//! Glues together language detection, streaming STT, post-processing,
//! and native text injection into a coherent session lifecycle.
//!
//! The pipeline runs in a background thread driven by `DictationCommand::Toggle`,
//! forwarding every `DictationEvent` directly to the caller-supplied `event_tx`
//! (the external UI consumer) — a single sender, no internal mirror channel.

mod pipeline;
#[cfg(test)]
pub(crate) mod test_support;

use std::sync::Arc;
use std::thread;

use crossbeam_channel::{unbounded, Sender};
use log::info;
pub use pipeline::DictationPipeline;
use vuho_domain::{DictationCommand, DictationEvent};
use vuho_os_integration::OsError;
use vuho_settings::SettingsStore;
use vuho_stt_engine::TranscriptionEngine;

/// Delivers cleaned text into the focused app (⌘V synthesis, falling back
/// to clipboard-only on failure — see `vuho_os_integration::inject_text`'s
/// own doc comment for the fallback policy).
///
/// Injected at [`DictationSession::new`] rather than called directly —
/// production passes `Arc::new(vuho_os_integration::inject_text)`, tests
/// pass a fake that records calls without touching CGEvent/clipboard APIs
/// (calling those from a non-main test thread crashes with SIGTRAP). This
/// replaces a `#[cfg(test)]`-gated static that used to switch behavior at
/// compile time instead of through the constructor.
pub type Injector = Arc<dyn Fn(&str) -> Result<(), OsError> + Send + Sync>;

/// Errors from sending a command to a [`DictationSession`].
///
/// Both variants mean the same practical thing to a caller (the command
/// did not reach the pipeline), but distinguish *why*: [`Self::SessionDropped`]
/// is a programming error (calling a session after dropping it, which
/// `Drop for DictationSession` prevents by construction since `self` can't
/// outlive its own drop — this variant exists for defense in depth, not
/// because it's reachable through the safe API today), while
/// [`Self::ChannelClosed`] is the pipeline thread having already exited
/// (e.g. during shutdown).
#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionError {
    /// The session's command channel was already torn down.
    #[error("session dropped")]
    SessionDropped,
    /// The command channel closed (the pipeline thread exited) before the
    /// command was delivered.
    #[error("command channel closed: pipeline thread has exited")]
    ChannelClosed,
}

/// A running dictation session.
///
/// Owns the streaming STT engine and drives the transcription pipeline in a
/// background thread. Post-processing and injection run inline on the
/// pipeline thread (the WP6 inverted-Stop fix already returns to Idle
/// immediately on Stop, so the command thread is never blocked).
pub struct DictationSession {
    /// Sends commands to the session (Toggle, Start, Stop).
    /// Wrapped in Option so Drop can take it to close the channel.
    command_tx: Option<Sender<DictationCommand>>,
    /// Handle to the background pipeline thread.
    handle: Option<thread::JoinHandle<()>>,
}

impl DictationSession {
    /// Creates a new dictation session.
    ///
    /// The session is not active until a `Toggle` (or `Start`) command is sent.
    /// Every `DictationEvent` the pipeline emits is forwarded to `event_tx`.
    ///
    /// `engine` must already be loaded — see `vuho_stt_engine::ParakeetEngine::load`.
    /// Taking it as a parameter keeps the multi-minute model load at app scope
    /// (CONSTITUTION rule 3) instead of on the command thread, where it would
    /// stall the first hotkey press and block every command behind it.
    ///
    /// `settings` is injected at construction (CONSTITUTION rule 5) — the
    /// pipeline reads the configured microphone device from it on every
    /// session start.
    ///
    /// `injector` performs the final ⌘V-or-clipboard delivery of post-processed
    /// text. Injected here — rather than the pipeline reaching for
    /// `vuho_os_integration::inject_text` directly, or a `#[cfg(test)]`
    /// static seam switching behavior at compile time — so production and
    /// tests drive the identical code path (CONSTITUTION rule 5); tests pass
    /// a fake that never touches CGEvent/clipboard APIs (which crash with
    /// SIGTRAP off the main thread), production passes
    /// `Arc::new(vuho_os_integration::inject_text)`.
    #[must_use]
    pub fn new(
        event_tx: Sender<DictationEvent>,
        engine: Box<dyn TranscriptionEngine + Send>,
        settings: Arc<SettingsStore>,
        injector: Injector,
    ) -> Self {
        let (command_tx, command_rx) = unbounded();

        let handle = thread::spawn(move || {
            info!("pipeline: thread started");
            let mut pipeline = DictationPipeline::new(
                command_rx,
                event_tx,
                engine,
                settings,
                pipeline::detect_input_language,
                injector,
            );
            pipeline.run();
            info!("pipeline: thread exiting");
        });

        Self {
            command_tx: Some(command_tx),
            handle: Some(handle),
        }
    }

    /// Send a command to the session.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] if the command channel is closed.
    pub fn command(&self, cmd: DictationCommand) -> Result<(), SessionError> {
        info!("session: command {cmd:?} → pipeline");
        self.command_tx
            .as_ref()
            .ok_or(SessionError::SessionDropped)?
            .send(cmd)
            .map_err(|_| SessionError::ChannelClosed)
    }

    /// Toggle the dictation session: start if idle, stop if recording.
    ///
    /// This is the primary entry point for the hotkey handler.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] if the command channel is closed.
    pub fn toggle(&self) -> Result<(), SessionError> {
        self.command(DictationCommand::Toggle)
    }

    /// Start a dictation session.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] if the command channel is closed.
    pub fn start(&self) -> Result<(), SessionError> {
        self.command(DictationCommand::Start)
    }

    /// Stop the current dictation session.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError`] if the command channel is closed.
    pub fn stop(&self) -> Result<(), SessionError> {
        self.command(DictationCommand::Stop)
    }
}

impl Drop for DictationSession {
    fn drop(&mut self) {
        // Send a best-effort Stop to finalize if mid-recording.
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(DictationCommand::Stop);
        }
        // Drop the command sender to close the channel → pipeline thread exits.
        self.command_tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vuho_domain::TranscriptionResult;
    use vuho_stt_engine::EngineError;

    use crate::test_support::fake_injector;

    fn empty_result() -> TranscriptionResult {
        TranscriptionResult {
            segments: Vec::new(),
            full_text: String::new(),
            language: "en".to_string(),
        }
    }

    /// Minimal ready engine: these tests exercise session plumbing, not STT,
    /// and must never load the real 1.5 GB model or touch a microphone.
    struct StubEngine;

    impl TranscriptionEngine for StubEngine {
        fn transcribe(
            &self,
            _samples: &[f32],
            _language: Option<&str>,
        ) -> Result<TranscriptionResult, EngineError> {
            Ok(empty_result())
        }
        fn unload(&self) {}
        fn start_stream(
            &self,
            _language: Option<&str>,
            _input_device: Option<&str>,
        ) -> Result<crossbeam_channel::Receiver<DictationEvent>, EngineError> {
            let (_tx, rx) = unbounded();
            Ok(rx)
        }
        fn stop_stream(&self) -> Result<TranscriptionResult, EngineError> {
            Ok(empty_result())
        }
    }

    /// A fresh `SettingsStore` backed by a per-test temp path — never
    /// touches the real `~/.config/vuho/settings.json`.
    fn test_settings() -> Arc<SettingsStore> {
        Arc::new(SettingsStore::new_temp("session"))
    }

    fn test_session(tx: Sender<DictationEvent>) -> DictationSession {
        let (injector, _received) = fake_injector();
        DictationSession::new(tx, Box::new(StubEngine), test_settings(), injector)
    }

    #[test]
    fn session_can_be_created() {
        let (tx, rx) = unbounded();
        let session = test_session(tx);
        // No session has been started, so nothing should arrive on the
        // caller's own receiver yet.
        assert!(rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err());
        // Send Stop to unblock the pipeline thread before dropping.
        session.stop().ok();
    }

    #[test]
    fn session_receives_stop_command() {
        let (tx, _rx) = unbounded();
        let session = test_session(tx);
        session.stop().unwrap();
        // The internal pipeline receives the stop and shuts down cleanly.
    }
}
