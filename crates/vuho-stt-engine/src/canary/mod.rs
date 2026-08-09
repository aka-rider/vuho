//! Canary-1B-v2: an attention encoder-decoder backend behind the shared
//! [`crate::window_inference::WindowInference`] seam.
//!
//! Unlike Parakeet-TDT, Canary decodes autoregressively from a language
//! prompt with no per-token acoustic alignment and no KV cache: each step
//! resubmits the whole `[1, S]` token tensor. It therefore supplies
//! *synthetic* token positions (a fixed stride) rather than measured
//! encoder frames — see [`models::CanaryModels`]'s `WindowInference` impl.

pub(crate) mod aed;
pub(crate) mod models;
pub mod prompt;

/// The id of the manifest's Canary model, if it declares one.
///
/// Found by backend rather than by name so no model id is written down in
/// this crate (ADR-019 — `models.manifest.json` is the one place ids live).
#[must_use]
pub fn manifest_model_id() -> Option<&'static str> {
    vuho_model_paths::manifest()
        .stt
        .models
        .iter()
        .find(|(_, model)| model.backend == vuho_model_paths::Backend::CanaryAed)
        .map(|(id, _)| id.as_str())
}
