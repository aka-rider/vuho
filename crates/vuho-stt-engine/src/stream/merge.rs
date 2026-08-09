//! Merge fresh tokens into committed tokens with overlap dedup.
//!
//! Adapted from `FluidAudio`'s `ChunkProcessor.swift`
//! `mergeChunks`/`findContiguousMatches`/`collapseSeamWordDuplicates`:
//! since decoding now always starts from a *fresh* `DecoderState` (see
//! `engine.rs`'s batch window loop and `stream/session.rs`'s design note —
//! carrying decoder state across independent windows is exactly what
//! caused the characterized blank-lock content-drop bug), two overlapping
//! windows/decodes independently re-transcribe the same physically
//! overlapping audio. Reconciling *that* — two probably-correct, possibly
//! differently-tokenized transcriptions of the same audio — is a
//! genuinely different (and much more tractable) problem than recovering
//! content a stale decoder state silently dropped.
//!
//! The two transcriptions of an overlap are usually near-identical but not
//! always *token*-aligned: an independent decode can choose a different
//! subword split for the same word (`"▁a"+"sk"` vs `"▁as"+"k"`, both
//! "ask"), and independent decodes routinely disagree on seam-word
//! capitalization (no shared sentence-start context) and trailing
//! punctuation (each decode guesses comma vs. period from only its own
//! audio, and audio that legitimately ends mid-phrase at a window/endpoint
//! boundary tends to attract a stray terminal period that isn't really
//! there). Matching individual tokens — even case/punctuation-normalized —
//! cannot see through a *different split*: neither `"a"` nor `"sk"` equals
//! `"as"` or `"k"` alone. So, mirroring `collapseSeamWordDuplicates`'s word
//! segmentation, the matcher first groups each side's tokens into **words**
//! (a word-initial token followed by its subword continuations and any
//! trailing punctuation), then compares whole reconstructed word text —
//! case-folded, edge punctuation stripped — rather than raw token ids.
//!
//! Once a match is found, the matched region (and everything after it) is
//! spliced in **from `fresh`, not `committed`** — `committed`'s own copy of
//! the matched words is discarded along with anything glued onto them
//! (like a stray terminal period): `fresh`'s decode had more trailing
//! context to resolve that boundary with, so its version of the seam is
//! the more reliable one. This is why `merge` returns how much of
//! `committed` to keep, not just a suffix to append — a deliberate
//! departure from a strict "never mutate committed, only append" contract.

use crate::token::TokenAt;

/// Result of [`merge`]: how much of `committed` the caller should keep
/// (`committed.truncate(keep_committed)`), followed by the tokens to
/// append (`committed.extend(append)`).
///
/// `pub` (not `pub(crate)`): re-exported by `bench_support` for
/// `benches/hot_paths.rs` — fields stay `pub(crate)`.
pub struct MergeOutcome {
    /// Number of leading `committed` tokens to keep.
    pub(crate) keep_committed: usize,
    /// Tokens to append after truncating `committed` to `keep_committed`.
    pub(crate) append: Vec<TokenAt>,
}

/// How far a backend's positions may be trusted when reconciling an
/// overlap: `search` bounds which tokens on each side are considered at
/// all, and `tolerance` is how far apart two otherwise-identical words'
/// positions may be and still count as the same word.
///
/// One overloaded `overlap_frames` parameter used to serve both roles.
/// They are independent: a backend whose positions are a fixed synthetic
/// stride rather than a measured frame index has meaningful ordering but
/// no meaningful distance, so it widens `tolerance` without widening
/// `search`.
///
/// `pub` (not `pub(crate)`): re-exported by `bench_support` for
/// `benches/hot_paths.rs`.
#[derive(Debug, Clone, Copy)]
pub struct MergeBounds {
    /// How far from the seam, in position units, to look for a match.
    pub search: usize,
    /// Maximum position difference between two matching words.
    pub tolerance: usize,
}

/// Merge `fresh` tokens into `committed`, reconciling the overlap.
///
/// `pub` (not `pub(crate)`): re-exported by `bench_support` for
/// `benches/hot_paths.rs`.
///
/// `bounds` comes from the backend ([`crate::window_inference::WindowInference::merge_bounds`]).
/// `piece(id)` returns `None` for a token with no vocabulary entry
/// (including blank), and `Some((is_word_initial, raw_text))` otherwise.
///
/// Algorithm:
/// 1. Restrict the search to each side's tokens within `bounds.search` of
///    the overlap — bounds the search to the physically-overlapping region
///    instead of the whole history. `committed`'s side is anchored at the
///    seam (its last position); `fresh`'s is anchored at the later of the
///    seam and `fresh`'s own first position, since the overlap is the head
///    of `fresh`'s window wherever the seam happens to sit.
/// 2. Segment each windowed slice into words (see `segment_words`, private).
/// 3. Find the longest contiguous run of words whose normalized core text
///    matches (case-folded, edge punctuation stripped) AND whose first
///    token's position is within `bounds.tolerance`, searched freely (not
///    anchored to either side's boundary).
/// 4. If that run has at least 2 words: keep `committed` only up to the
///    start of the matched region, then append `fresh` from that same
///    matched region onward (using `fresh`'s copy of the seam).
/// 5. Otherwise fall back to keeping all of `committed` and dropping every
///    `fresh` token whose position is at or before the last committed
///    position.
pub fn merge<'p>(
    committed: &[TokenAt],
    fresh: Vec<TokenAt>,
    bounds: MergeBounds,
    piece: impl Fn(u32) -> Option<(bool, &'p str)>,
) -> MergeOutcome {
    if fresh.is_empty() || committed.is_empty() {
        return MergeOutcome {
            keep_committed: committed.len(),
            append: fresh,
        };
    }

    let boundary_pos = committed.last().map_or(0, |t| t.pos);

    let overlap_committed_start =
        committed.partition_point(|t| t.pos.saturating_add(bounds.search) < boundary_pos);
    let overlap_committed = &committed[overlap_committed_start..];
    // The fresh side's overlap is the head of `fresh`'s own window, so the
    // bound is anchored at whichever is later: the seam, or `fresh`'s first
    // position. Anchoring on the seam alone silently empties this slice
    // whenever `committed`'s last position lands *before* `fresh` even
    // starts — which a backend with estimated positions can produce (a
    // sparse window's tokens are placed conservatively early, see
    // `canary::models::stamp_positions`), and an empty fresh slice means no
    // match is possible and the whole overlap is transcribed twice.
    // Measured positions never trigger the `max`, so this leaves Parakeet
    // exactly as it was.
    let fresh_search_end = boundary_pos
        .max(fresh.first().map_or(0, |t| t.pos))
        .saturating_add(bounds.search);
    let overlap_fresh_end = fresh
        .iter()
        .position(|t| t.pos > fresh_search_end)
        .unwrap_or(fresh.len());
    let overlap_fresh = &fresh[..overlap_fresh_end];

    let committed_words = segment_words(overlap_committed, &piece);
    let fresh_words = segment_words(overlap_fresh, &piece);

    if let Some((committed_word_start, fresh_word_start, len)) =
        longest_contiguous_word_run(&committed_words, &fresh_words, bounds.tolerance)
    {
        if len >= 2 {
            let keep_committed =
                overlap_committed_start + committed_words[committed_word_start].token_start;
            let append_from = fresh_words[fresh_word_start].token_start;
            return MergeOutcome {
                keep_committed,
                append: fresh[append_from..].to_vec(),
            };
        }
    }

    let drop_count = fresh.iter().take_while(|t| t.pos <= boundary_pos).count();
    MergeOutcome {
        keep_committed: committed.len(),
        append: fresh[drop_count..].to_vec(),
    }
}

/// One word: the token-index span `[token_start, token_end)` into the
/// slice it was segmented from, the position of its first token, and its
/// normalized core text (all constituent tokens' raw text concatenated,
/// lowercased, with leading/trailing non-alphanumeric characters
/// stripped).
struct Word {
    token_start: usize,
    token_end: usize,
    pos: usize,
    core: String,
}

/// Group `tokens` into words: each word-initial token (per `piece`) starts
/// a new word; every other token (subword continuations, and punctuation
/// that never starts its own word) attaches to the word already open. The
/// very first token always starts a word regardless of `is_word_initial`,
/// so a slice that begins mid-word still segments into *some* word rather
/// than being dropped.
fn segment_words<'p>(
    tokens: &[TokenAt],
    piece: &impl Fn(u32) -> Option<(bool, &'p str)>,
) -> Vec<Word> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut starts = vec![0usize];
    for (i, t) in tokens.iter().enumerate().skip(1) {
        if piece(t.id).is_some_and(|(is_word_initial, _)| is_word_initial) {
            starts.push(i);
        }
    }

    let words: Vec<Word> = starts
        .iter()
        .enumerate()
        .map(|(idx, &start)| {
            let end = starts.get(idx + 1).copied().unwrap_or(tokens.len());
            build_word(tokens, start, end, piece)
        })
        .collect();

    debug_assert_eq!(words.first().map(|w| w.token_start), Some(0));
    debug_assert_eq!(words.last().map(|w| w.token_end), Some(tokens.len()));
    debug_assert!(words
        .windows(2)
        .all(|pair| pair[0].token_end == pair[1].token_start));

    words
}

/// Build a [`Word`] spanning `tokens[start..end]`.
fn build_word<'p>(
    tokens: &[TokenAt],
    start: usize,
    end: usize,
    piece: &impl Fn(u32) -> Option<(bool, &'p str)>,
) -> Word {
    let mut raw = String::new();
    for t in &tokens[start..end] {
        if let Some((_, text)) = piece(t.id) {
            raw.push_str(text);
        }
    }
    let core = raw
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    Word {
        token_start: start,
        token_end: end,
        pos: tokens[start].pos,
        core,
    }
}

/// Find the longest contiguous run of matching words between `left` and
/// `right`, searched freely (any starting offset in either slice) rather
/// than anchored to a boundary. Two words match when both have a
/// non-empty core, the cores are equal, and their positions are within
/// `tolerance`. Returns `(left_start, right_start, length)` for the best
/// run found, or `None` if no pair matches at all.
///
/// `left`/`right` are expected to be small (bounded to one overlap
/// window's worth of words), so the O(n·m) scan is not a concern.
fn longest_contiguous_word_run(
    left: &[Word],
    right: &[Word],
    tolerance: usize,
) -> Option<(usize, usize, usize)> {
    let matches = |a: &Word, b: &Word| {
        !a.core.is_empty() && a.core == b.core && a.pos.abs_diff(b.pos) <= tolerance
    };

    let mut best: Option<(usize, usize, usize)> = None;
    for i in 0..left.len() {
        for j in 0..right.len() {
            if !matches(&left[i], &right[j]) {
                continue;
            }
            let mut len = 0;
            while i + len < left.len()
                && j + len < right.len()
                && matches(&left[i + len], &right[j + len])
            {
                len += 1;
            }
            if best.is_none_or(|(_, _, best_len)| len > best_len) {
                best = Some((i, j, len));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(id: u32, pos: usize) -> TokenAt {
        TokenAt { id, pos }
    }

    /// The bounds Parakeet supplies (`OVERLAP_FRAMES = 25`), i.e. what
    /// every one of these cases used to pass as a bare `25`.
    fn bounds(search: usize) -> MergeBounds {
        MergeBounds {
            search,
            tolerance: search / 2,
        }
    }

    /// Test vocabulary: every id is its own whole word (`is_word_initial`
    /// always true), piece text is `format!(" w{id}")`. Id `0` resolves to
    /// `None` (simulates blank / an unknown id). Leaks the formatted string
    /// to satisfy `piece_info`'s real `&str`-borrowing signature (WP9) —
    /// test-only, a handful of calls per test run, never a hot path.
    fn word_piece(id: u32) -> Option<(bool, &'static str)> {
        if id == 0 {
            return None;
        }
        let leaked: &'static str = Box::leak(format!(" w{id}").into_boxed_str());
        Some((true, leaked))
    }

    #[test]
    fn exact_duplicate_overlap_drops_prefix() {
        let committed = vec![tok(1, 10), tok(2, 11), tok(3, 12)];
        let fresh = vec![tok(1, 10), tok(2, 11), tok(3, 12), tok(4, 13), tok(5, 14)];
        let result = merge(&committed, fresh, bounds(25), word_piece);

        assert_eq!(result.keep_committed, 0);
        assert_eq!(
            result.append,
            vec![tok(1, 10), tok(2, 11), tok(3, 12), tok(4, 13), tok(5, 14)]
        );
    }

    #[test]
    fn word_disagreement_falls_back_to_position_drop() {
        let committed = vec![tok(1, 10), tok(2, 11), tok(3, 12)];
        let fresh = vec![tok(99, 12), tok(4, 13), tok(5, 14)];
        let result = merge(&committed, fresh, bounds(25), word_piece);

        assert_eq!(result.keep_committed, committed.len());
        assert_eq!(result.append, vec![tok(4, 13), tok(5, 14)]);
    }

    #[test]
    fn no_overlap_keeps_all_fresh() {
        let committed = vec![tok(1, 10)];
        let fresh = vec![tok(2, 20), tok(3, 21)];
        let result = merge(&committed, fresh.clone(), bounds(25), word_piece);
        assert_eq!(result.keep_committed, committed.len());
        assert_eq!(result.append, fresh);
    }

    #[test]
    fn empty_committed_returns_all_fresh() {
        let fresh = vec![tok(1, 10), tok(2, 11)];
        let result = merge(&[], fresh.clone(), bounds(25), word_piece);
        assert_eq!(result.keep_committed, 0);
        assert_eq!(result.append, fresh);
    }

    #[test]
    fn empty_fresh_returns_empty() {
        let committed = vec![tok(1, 10)];
        let result = merge(&committed, vec![], bounds(25), word_piece);
        assert_eq!(result.keep_committed, committed.len());
        assert!(result.append.is_empty());
    }

    #[test]
    fn idempotence_self_merge() {
        let tokens = vec![tok(1, 10), tok(2, 11), tok(3, 12)];
        let result = merge(&tokens, tokens.clone(), bounds(25), word_piece);
        assert_eq!(result.keep_committed, 0);
        assert_eq!(result.append, tokens);
    }

    #[test]
    fn all_fresh_behind_committed() {
        let committed = vec![tok(1, 100)];
        let fresh = vec![tok(2, 50), tok(3, 60)];
        let result = merge(&committed, fresh, bounds(25), word_piece);
        assert_eq!(result.keep_committed, committed.len());
        assert!(result.append.is_empty());
    }

    #[test]
    fn single_word_match_falls_back() {
        let committed = vec![tok(1, 10), tok(2, 11)];
        let fresh = vec![tok(1, 10), tok(3, 15)];
        let result = merge(&committed, fresh, bounds(25), word_piece);

        assert_eq!(result.keep_committed, committed.len());
        assert_eq!(result.append, vec![tok(3, 15)]);
    }

    /// A match must be found even when `fresh` has one extra, non-matching
    /// word before the matching run starts (not anchored to `fresh`'s
    /// head), and `committed` is truncated back to before its own copy of
    /// the matched words.
    #[test]
    fn match_not_anchored_at_fresh_head_is_still_found() {
        let committed = vec![tok(1, 10), tok(2, 11), tok(3, 12)];
        let fresh = vec![tok(77, 9), tok(2, 11), tok(3, 12), tok(4, 13)];
        let result = merge(&committed, fresh, bounds(25), word_piece);

        assert_eq!(result.keep_committed, 1);
        assert_eq!(result.append, vec![tok(2, 11), tok(3, 12), tok(4, 13)]);
    }

    #[test]
    fn match_found_within_overlap_window_not_just_at_tail() {
        let committed = vec![tok(1, 5), tok(2, 6), tok(3, 7), tok(4, 8)];
        let fresh = vec![tok(3, 7), tok(4, 8), tok(5, 9)];
        let result = merge(&committed, fresh, bounds(4), word_piece);

        assert_eq!(result.keep_committed, 2);
        assert_eq!(result.append, vec![tok(3, 7), tok(4, 8), tok(5, 9)]);
    }

    /// Two independently-decoded overlaps split the *same word* into
    /// different subword pieces ("ask" as "▁a"+"sk" on one side, "▁as"+"k"
    /// on the other) — per-token matching cannot see this as a duplicate,
    /// but word-level matching (comparing the concatenated, normalized
    /// text) does.
    #[test]
    fn different_subword_split_of_the_same_word_still_matches() {
        let committed = vec![tok(10, 10), tok(11, 10)];
        let fresh = vec![tok(20, 10), tok(21, 10), tok(30, 14)];

        let piece = |id: u32| -> Option<(bool, &str)> {
            match id {
                10 => Some((true, " a")),
                11 => Some((false, "sk")),
                20 => Some((true, " as")),
                21 => Some((false, "k")),
                30 => Some((true, " new")),
                _ => None,
            }
        };

        let committed_words = segment_words(&committed, &piece);
        let fresh_words = segment_words(&fresh, &piece);
        assert_eq!(committed_words[0].core, "ask");
        assert_eq!(fresh_words[0].core, "ask");

        let best = longest_contiguous_word_run(&committed_words, &fresh_words, 25);
        assert_eq!(
            best,
            Some((0, 0, 1)),
            "the differently-split word must still be recognized as the same word"
        );
    }

    /// Case and trailing-punctuation disagreement at the seam does not
    /// defeat the match, and the re-spliced result uses `fresh`'s copy of
    /// the seam words.
    #[test]
    fn case_and_punctuation_disagreement_does_not_defeat_match() {
        let committed = vec![tok(1, 10), tok(2, 11), tok(3, 12)];
        let fresh = vec![tok(101, 10), tok(102, 11), tok(200, 15)];

        let piece = |id: u32| -> Option<(bool, &str)> {
            match id {
                1 => Some((true, " Ask")),
                2 => Some((true, " not,")),
                3 => Some((true, " what")),
                101 => Some((true, " ask")),
                102 => Some((true, " not.")),
                200 => Some((true, " new")),
                _ => None,
            }
        };

        let result = merge(&committed, fresh, bounds(25), piece);
        assert_eq!(result.keep_committed, 0);
        assert_eq!(
            result.append,
            vec![tok(101, 10), tok(102, 11), tok(200, 15)]
        );
    }

    #[test]
    fn longest_contiguous_word_run_prefers_the_longest_match() {
        let mk = |core: &str, pos: usize| Word {
            token_start: 0,
            token_end: 1,
            pos,
            core: core.to_string(),
        };
        let left = vec![mk("nine", 0), mk("one", 1), mk("two", 2), mk("three", 3)];
        let right = vec![
            mk("nine", 0),
            mk("one", 1),
            mk("two", 2),
            mk("three", 3),
            mk("four", 4),
        ];

        let best = longest_contiguous_word_run(&left, &right, 0);
        assert_eq!(best, Some((0, 0, 4)));
    }

    #[test]
    fn longest_contiguous_word_run_none_when_no_match() {
        let mk = |core: &str, pos: usize| Word {
            token_start: 0,
            token_end: 1,
            pos,
            core: core.to_string(),
        };
        let left = vec![mk("one", 0), mk("two", 1)];
        let right = vec![mk("three", 0), mk("four", 1)];
        assert_eq!(longest_contiguous_word_run(&left, &right, 0), None);
    }

    #[test]
    fn segment_words_folds_trailing_punctuation_into_preceding_word() {
        let tokens = vec![tok(1, 0), tok(2, 1)];
        let piece = |id: u32| -> Option<(bool, &str)> {
            match id {
                1 => Some((true, " hello")),
                2 => Some((false, ",")),
                _ => None,
            }
        };
        let words = segment_words(&tokens, &piece);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].core, "hello");
        assert_eq!(words[0].token_start, 0);
        assert_eq!(words[0].token_end, 2);
    }

    #[test]
    fn segment_words_empty_slice() {
        assert!(segment_words(&[], &word_piece).is_empty());
    }

    /// A backend with estimated positions can place a sparse window's
    /// tokens conservatively early, leaving `committed`'s last position
    /// before `fresh`'s first. The fresh side of the overlap must still be
    /// searched — anchoring its bound on the seam alone would empty the
    /// slice, and an empty slice can match nothing, so the whole overlap
    /// would be appended a second time.
    #[test]
    fn a_seam_behind_fresh_still_searches_freshs_head() {
        let committed = vec![tok(1, 0), tok(2, 4), tok(3, 8)];
        let fresh = vec![tok(2, 100), tok(3, 104), tok(4, 108)];
        // Estimated positions come with `tolerance: usize::MAX` (text-only
        // matching) — the proximity gate would reject this seam regardless
        // of the search bound.
        let estimated = MergeBounds {
            search: 25,
            tolerance: usize::MAX,
        };
        let result = merge(&committed, fresh, estimated, word_piece);

        assert_eq!(
            result.keep_committed, 1,
            "the seam words must be found in fresh's head and re-spliced"
        );
        assert_eq!(result.append, vec![tok(2, 100), tok(3, 104), tok(4, 108)]);
    }

    /// A backend whose positions are a synthetic stride passes `usize::MAX`
    /// for both bounds to mean "match on text alone, unbounded". Plain `+`
    /// overflowed and panicked in debug builds; `saturating_add` makes both
    /// search predicates degenerate to "consider the whole slice", which is
    /// exactly the intended semantics.
    #[test]
    fn unbounded_search_and_tolerance_do_not_overflow() {
        let unbounded = MergeBounds {
            search: usize::MAX,
            tolerance: usize::MAX,
        };

        let committed = vec![tok(1, 10), tok(2, 11), tok(3, 12)];
        let fresh = vec![tok(2, 900), tok(3, 904), tok(4, 908)];
        let result = merge(&committed, fresh, unbounded, word_piece);

        assert_eq!(
            result.keep_committed, 1,
            "an unbounded search must still find the seam and re-splice from fresh"
        );
        assert_eq!(result.append, vec![tok(2, 900), tok(3, 904), tok(4, 908)]);
    }

    /// Unbounded bounds must also survive the no-match fallback path, where
    /// an overflowing search bound panicked before any match could even be
    /// attempted — positions near `usize::MAX` make that overflow certain.
    #[test]
    fn unbounded_bounds_survive_the_no_match_fallback() {
        let unbounded = MergeBounds {
            search: usize::MAX,
            tolerance: usize::MAX,
        };

        let committed = vec![tok(1, usize::MAX - 1)];
        let fresh = vec![tok(9, usize::MAX)];
        let result = merge(&committed, fresh.clone(), unbounded, word_piece);

        assert_eq!(result.keep_committed, committed.len());
        assert_eq!(result.append, fresh);
    }
}
