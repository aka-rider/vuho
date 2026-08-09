//! The committed-transcript accumulator: `committed` tokens, `segments`,
//! and `segment_id` bookkeeping, in exactly one place (CONSTITUTION rule
//! 26) shared by batch `transcribe()` (`engine.rs`) and the streaming
//! session (`stream::session`) — before this module existed, both kept
//! their own near-identical copy of "truncate committed to
//! `keep_committed`, then push a segment and extend" (engine.rs's window
//! loop and `SessionState::apply_merge`/`push_segment`).
//!
//! # `segments` vs. `full_text`: the truncation trade-off
//!
//! [`Accumulator::apply`] takes a [`MergeOutcome`], whose `keep_committed`
//! can be *less* than `committed`'s current length — `stream::merge`
//! deliberately re-splices a matched seam using the newer, higher-context
//! decode's copy of the words (see `stream::merge`'s doc comment), which
//! means `committed`, and therefore what a later `full_text()` call
//! detokenizes, can retroactively change at a seam that an *earlier*
//! `apply` call already turned into a pushed [`TranscriptSegment`].
//!
//! `segments` is never retroactively edited or removed when this happens:
//! it is informational per-window/per-commit metadata (timestamps + the
//! text as decoded *at that time*), not the correctness gate. `full_text()`
//! — built fresh from `committed` on demand — is the correctness gate: it
//! always reflects every seam correction `apply`/`promote` has folded in,
//! even for a seam whose already-pushed segment text has since drifted out
//! of exact sync with it. Reconciling that drift retroactively (rewriting
//! or dropping old segments) was considered and rejected: segments are
//! purely a UI/inspection aid (per-chunk timing), so a rare, small
//! trailing-word mismatch there is an accepted trade-off rather than
//! something worth the complexity of retroactive segment editing.

use vuho_domain::TranscriptSegment;

use crate::token::{frame_ms, TokenAt};
use crate::window_inference::WindowInference;

use super::merge::MergeOutcome;

/// Owns the confirmed transcript state: committed tokens, the segments
/// derived from them, and the next segment id.
pub(crate) struct Accumulator {
    /// All committed (confirmed) tokens, session-global-position indexed.
    committed: Vec<TokenAt>,
    segments: Vec<TranscriptSegment>,
    segment_id: u32,
}

impl Accumulator {
    pub(crate) fn new() -> Self {
        Self {
            committed: Vec::new(),
            segments: Vec::new(),
            segment_id: 0,
        }
    }

    /// The committed tokens so far — the anchor `stream::merge::merge`
    /// reconciles new decodes against.
    pub(crate) fn committed(&self) -> &[TokenAt] {
        &self.committed
    }

    /// Apply a `merge::MergeOutcome`: truncate `committed` to
    /// `keep_committed`, then, if `append` is non-empty, push a
    /// [`TranscriptSegment`] for it and extend `committed` — the single
    /// home of this logic (see this module's doc comment for the
    /// `segments`/`full_text` trade-off it implies).
    pub(crate) fn apply(&mut self, outcome: MergeOutcome, models: &dyn WindowInference) {
        self.committed.truncate(outcome.keep_committed);
        if outcome.append.is_empty() {
            return;
        }
        self.push_segment(&outcome.append, models);
        self.committed.extend(outcome.append);
    }

    /// Build a `TranscriptSegment` for a run of newly committed tokens and
    /// push it, advancing `segment_id`.
    ///
    /// The segment's `start_ms`/`end_ms` are only as meaningful as the
    /// backend's `TokenAt::pos` (opaque and possibly synthetic — see
    /// [`TokenAt`]): informational, never a correctness gate, and read
    /// only by `test-stt-ffi`'s diagnostic printout.
    fn push_segment(&mut self, tokens: &[TokenAt], models: &dyn WindowInference) {
        let (Some(first), Some(last)) = (tokens.first(), tokens.last()) else {
            return;
        };
        let text = models.detokenize(tokens);
        self.segments.push(TranscriptSegment::new(
            self.segment_id,
            text,
            frame_ms(first.pos),
            frame_ms(last.pos),
        ));
        self.segment_id += 1;
    }

    /// Detokenized `committed` — the correctness gate (see this module's
    /// doc comment).
    pub(crate) fn full_text(&self, models: &dyn WindowInference) -> String {
        models.detokenize(&self.committed)
    }

    /// `(confirmed, unconfirmed)` detokenized — the producer-supplied pair a
    /// streaming `PartialTranscript` event needs directly (ADR-018): a
    /// `(String, String)` from `committed` and the caller's own
    /// not-yet-committed ("fresh") token run, with no concatenation of the
    /// two token vectors (the previous `texts_with` cloned the whole
    /// `committed` vec on every partial just to detokenize a combined
    /// prefix the caller immediately had to subtract `unconfirmed_text`
    /// back out of again downstream).
    pub(crate) fn confirmed_unconfirmed_texts(
        &self,
        extra: &[TokenAt],
        models: &dyn WindowInference,
    ) -> (String, String) {
        (models.detokenize(&self.committed), models.detokenize(extra))
    }

    /// Consume the accumulator, returning its segments (for the final
    /// `TranscriptionResult`).
    pub(crate) fn into_segments(self) -> Vec<TranscriptSegment> {
        self.segments
    }
}
