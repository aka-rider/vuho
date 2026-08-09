//! The seam every STT backend implements for the shared streaming
//! pipeline (`stream::session`, `stream::accumulator`, `stream::merge`).
//!
//! Deliberately **not** `Send + Sync`: `CoreML`'s `MLModel` handles are
//! neither, which is why `coreml::SendModel` exists as a per-concrete-type
//! wrapper. A supertrait here would not compile at the implementors, and
//! is unnecessary — a backend crosses a thread boundary as
//! `Arc<SendModel<M>>`, and the `&dyn WindowInference` is only ever formed
//! inside the session thread that already owns it.

use crate::stream::merge::MergeBounds;
use crate::token::TokenAt;
use crate::EngineError;

/// One decode of one audio window, plus the vocabulary and position
/// semantics the shared pipeline needs to reconcile consecutive windows.
pub(crate) trait WindowInference {
    /// Decode `samples` (one window's worth, zero-padded internally to
    /// whatever fixed length the backend's models require) into tokens
    /// whose `pos` is offset by `global_frame_offset`.
    ///
    /// `language` is a BCP-47-derived code; a backend that decodes without
    /// a language prompt ignores it.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::CoreMl` if a `CoreML` call fails, or
    /// `EngineError::Transcribe` if a decode invariant is violated.
    fn infer_window(
        &self,
        samples: &[f32],
        global_frame_offset: usize,
        language: &str,
    ) -> Result<Vec<TokenAt>, EngineError>;

    /// `(is_word_initial, raw_piece_text)` for a token id, or `None` for an
    /// id with no vocabulary entry.
    fn piece_info(&self, id: u32) -> Option<(bool, &str)>;

    /// Render tokens to text using this backend's vocabulary.
    fn detokenize(&self, tokens: &[TokenAt]) -> String;

    /// How far this backend's `TokenAt::pos` values may be trusted when
    /// reconciling an overlap.
    fn merge_bounds(&self) -> MergeBounds;
}
