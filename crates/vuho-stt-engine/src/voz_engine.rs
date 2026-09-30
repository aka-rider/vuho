//! The voz transcription engine.
//!
//! App-scoped, loaded once (CONSTITUTION rule 3). A thin wrapper around
//! [`StreamingEngine`] — everything backend-independent lives there; this
//! module supplies only the voz-specific load step. Voz decodes without a
//! language prompt (like Parakeet, it takes any language the model was
//! trained on), so there is no pre-capture language check.

use std::path::PathBuf;

use crossbeam_channel::Receiver;
use vuho_domain::{DictationEvent, TranscriptionResult};

use crate::coreml::SendModel;
use crate::streaming_engine::StreamingEngine;
use crate::voz::models::VozModels;
use crate::EngineError;

/// The voz STT engine.
pub struct VozEngine(StreamingEngine<SendModel<VozModels>>);

impl VozEngine {
    /// Load the voz engine for `model_id` from a resolved model folder.
    ///
    /// Also warms the models on the calling thread — the first Neural
    /// Engine plan compilation can take tens of seconds, so callers that
    /// must not block should load off their command thread.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if any component fails to load or
    /// disagrees with `meta.json`, or `EngineError::UnknownModel` if
    /// `model_id` names no model in the embedded manifest.
    ///
    /// Takes `PathBuf` by value for the same reason `ParakeetEngine::load`
    /// does — it composes with `resolve_model_folder`'s result at call sites.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load(model_id: &str, model_folder: PathBuf) -> Result<Self, EngineError> {
        Ok(Self(StreamingEngine::new(SendModel(VozModels::load(
            model_id,
            &model_folder,
        )?))))
    }
}

impl crate::TranscriptionEngine for VozEngine {
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
