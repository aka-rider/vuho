//! The Canary-1B-v2 transcription engine.
//!
//! App-scoped, loaded once (CONSTITUTION rule 3). A thin wrapper around
//! [`StreamingEngine`] — everything backend-independent lives there; this
//! module supplies the Canary-specific load step and the pre-capture
//! language check.

use std::path::PathBuf;

use crossbeam_channel::Receiver;
use vuho_domain::{DictationEvent, TranscriptionResult};

use crate::canary::models::CanaryModels;
use crate::canary::prompt;
use crate::streaming_engine::{resolve_language, StreamingEngine};
use crate::EngineError;

/// The Canary-1B-v2 STT engine.
pub struct CanaryEngine {
    inner: StreamingEngine<CanaryModels>,
    /// The manifest display name, for the unsupported-language error raised
    /// before any model call happens.
    display_name: String,
}

impl CanaryEngine {
    /// Load the Canary engine for `model_id` from a resolved model folder.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if any component fails to load, or
    /// `EngineError::UnknownModel` if `model_id` names no model in the
    /// embedded manifest.
    ///
    /// Takes `PathBuf` by value for the same reason `ParakeetEngine::load`
    /// does — it composes with `resolve_model_folder`'s result at call sites.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load(model_id: &str, model_folder: PathBuf) -> Result<Self, EngineError> {
        let display_name = vuho_model_paths::manifest()
            .stt
            .model(model_id)
            .map_or_else(|| model_id.to_owned(), |m| m.display_name.clone());
        Ok(Self {
            inner: StreamingEngine::new(CanaryModels::load(model_id, &model_folder)?),
            display_name,
        })
    }

    /// Reject a language this backend cannot transcribe.
    ///
    /// Canary needs an explicit source-language prompt token; there is no
    /// auto-detect and no safe default (prompting `<|en|>` for a Japanese
    /// speaker would be a hidden failure, CONSTITUTION rule 2).
    fn check_language(&self, language: &str) -> Result<(), EngineError> {
        if prompt::transcribe_prompt(language).is_some() {
            return Ok(());
        }
        Err(EngineError::UnsupportedLanguage {
            model: self.display_name.clone(),
            language: language.to_owned(),
        })
    }
}

impl crate::TranscriptionEngine for CanaryEngine {
    fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, EngineError> {
        self.check_language(&resolve_language(language))?;
        self.inner.transcribe(samples, language)
    }

    fn unload(&self) {
        self.inner.unload();
    }

    /// Validates the language **before** starting capture, so an
    /// unsupported language never produces a session the caller would then
    /// announce as started (CONSTITUTION rule 11).
    fn start_stream(
        &self,
        language: Option<&str>,
        input_device: Option<&str>,
    ) -> Result<Receiver<DictationEvent>, EngineError> {
        self.check_language(&resolve_language(language))?;
        self.inner.start_stream(language, input_device)
    }

    fn stop_stream(&self) -> Result<TranscriptionResult, EngineError> {
        self.inner.stop_stream()
    }
}
