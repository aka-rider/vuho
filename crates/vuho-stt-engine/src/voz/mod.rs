//! Voz: Desert Ant Labs' re-export of Parakeet-TDT 0.6B v3 as three fp16
//! `CoreML` bundles behind the shared [`crate::window_inference::WindowInference`]
//! seam.
//!
//! Where Parakeet runs Preprocessor → Encoder → (Decoder + Joint) with an
//! `RNNTJoint` call per frame, voz folds the joint into the decoder and
//! projects the encoder output up front, so one decoder call scores eight
//! consecutive encoder frames. The frame axis is still the real 80 ms
//! encoder frame, so its token positions are measured, not estimated.

pub(crate) mod embedding;
pub(crate) mod frontend;
pub(crate) mod meta;
pub(crate) mod models;
pub(crate) mod tdt;

/// The id of the manifest's voz model, if it declares one.
///
/// Found by backend rather than by name so no model id is written down in
/// this crate (ADR-019 — `models.manifest.json` is the one place ids live).
#[must_use]
pub fn manifest_model_id() -> Option<&'static str> {
    vuho_model_paths::manifest()
        .stt
        .models
        .iter()
        .find(|(_, model)| model.backend == vuho_model_paths::Backend::VozTdt)
        .map(|(id, _)| id.as_str())
}
