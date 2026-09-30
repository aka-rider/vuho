//! The ONNX Parakeet-TDT transcription engine (Linux).
//!
//! App-scoped, loaded once (CONSTITUTION rule 3). A thin wrapper around
//! [`StreamingEngine`], like `ParakeetEngine`: this module supplies only the
//! load step.

use std::path::PathBuf;

use crossbeam_channel::Receiver;
use vuho_domain::{DictationEvent, TranscriptionResult};

use crate::parakeet::onnx_models::OnnxParakeetModels;
use crate::streaming_engine::StreamingEngine;
use crate::EngineError;

/// The Parakeet-TDT engine running the ONNX export on ONNX Runtime.
pub struct OnnxParakeetEngine(StreamingEngine<OnnxParakeetModels>);

impl OnnxParakeetEngine {
    /// Load the engine for `model_id` from a resolved model folder, and warm
    /// it up on the calling thread.
    ///
    /// # Errors
    ///
    /// `EngineError::LoadFailed` if the folder lacks an asset, a graph's
    /// signature is not what this backend feeds it, or the warm-up inference
    /// fails; `EngineError::Onnx` if ONNX Runtime rejects a graph;
    /// `EngineError::UnknownModel` if `model_id` names no model in the
    /// embedded manifest.
    ///
    /// Takes `PathBuf` by value for the same reason `ParakeetEngine::load`
    /// does — it composes with `resolve_model_folder`'s result at call sites.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load(model_id: &str, model_folder: PathBuf) -> Result<Self, EngineError> {
        Ok(Self(StreamingEngine::new(OnnxParakeetModels::load(
            model_id,
            &model_folder,
        )?)))
    }
}

impl crate::TranscriptionEngine for OnnxParakeetEngine {
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
