//! The backend-independent decoded-token type and the one frame→ms
//! conversion.
//!
//! Lives at crate level, not inside `parakeet`, because every backend
//! behind [`crate::window_inference::WindowInference`] emits these and the
//! shared streaming pipeline (`stream::merge`, `stream::accumulator`,
//! `stream::session`) consumes them.

/// Duration of one encoder frame in milliseconds (1280 samples @ 16kHz = 80ms).
const FRAME_MS: usize = 80;

/// Convert a token position to milliseconds.
///
/// The one place this conversion happens (CONSTITUTION rule 26) — both the
/// batch window loop (`engine.rs`) and the streaming accumulator
/// (`stream::accumulator`) call this rather than keeping their own copies.
///
/// `pub` (not `pub(crate)`): re-exported by `crate::bench_support`.
#[must_use]
pub fn frame_ms(pos: usize) -> u64 {
    (pos * FRAME_MS) as u64
}

/// A token emitted during decoding, with its position in the session.
///
/// `pub` (not `pub(crate)`): re-exported by `bench_support` for
/// `benches/hot_paths.rs` — fields stay `pub(crate)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenAt {
    /// Token id in the vocabulary.
    pub(crate) id: u32,
    /// Position, supplied by the backend, as a session-global 1280-sample
    /// encoder-frame index. `stream::merge` reads it as exactly that:
    /// `search` bounds the seam hunt to a frame count either side of the
    /// last committed position, and the no-match fallback drops every fresh
    /// token at or before it. Positions on any other axis mis-locate the
    /// overlap and either duplicate or delete real transcript, so a decoder
    /// with no acoustic alignment must still *estimate* a frame index —
    /// Canary spreads a window's tokens over the frames its own encoder
    /// reports (`canary::models::stamp_positions`) — rather than invent a
    /// per-token stride.
    pub(crate) pos: usize,
}
