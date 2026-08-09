//! The Parakeet-TDT transcription engine.
//!
//! App-scoped, loaded once at startup (CONSTITUTION rule 3). A thin wrapper
//! around [`StreamingEngine`], which owns everything backend-independent
//! (batch windowing, the streaming session lifecycle, the single live-session
//! slot); this module supplies only the Parakeet-specific load step.

use std::path::PathBuf;

use crossbeam_channel::Receiver;
use vuho_domain::{DictationEvent, TranscriptionResult};

use crate::parakeet::models::ParakeetModels;
use crate::streaming_engine::StreamingEngine;
use crate::EngineError;

/// The Parakeet-TDT STT engine, loaded once at app startup.
pub struct ParakeetEngine(StreamingEngine<ParakeetModels>);

impl ParakeetEngine {
    /// Load the Parakeet-TDT engine for `model_id` from a resolved model
    /// folder.
    ///
    /// The folder must contain every asset `model_id` declares in the
    /// embedded manifest (validated by [`crate::validate_model_layout`]
    /// during load). This also warms the models on the calling thread —
    /// callers that must not block should do this warmup off the command
    /// thread (as `vuho-ui`'s `spawn_warmup_and_bridge` does).
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if any model fails to load, or
    /// `EngineError::UnknownModel` if `model_id` names no model in the
    /// embedded manifest.
    ///
    /// Takes `PathBuf` by value (not `&Path`) so this composes directly with
    /// `resolve_model_folder(id).and_then(|f| ParakeetEngine::load(id, f))`
    /// at call sites (`vuho-ui`'s `spawn_warmup_and_bridge`, `test-stt-ffi`).
    #[allow(clippy::needless_pass_by_value)]
    pub fn load(model_id: &str, model_folder: PathBuf) -> Result<Self, EngineError> {
        Ok(Self(StreamingEngine::new(ParakeetModels::load(
            model_id,
            &model_folder,
        )?)))
    }
}

impl crate::TranscriptionEngine for ParakeetEngine {
    fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, EngineError> {
        self.0.transcribe(samples, language)
    }

    fn unload(&self) {
        self.0.unload();
    }

    fn start_stream(
        &self,
        language: Option<&str>,
        input_device: Option<&str>,
    ) -> Result<Receiver<DictationEvent>, EngineError> {
        self.0.start_stream(language, input_device)
    }

    fn stop_stream(&self) -> Result<TranscriptionResult, EngineError> {
        self.0.stop_stream()
    }
}
