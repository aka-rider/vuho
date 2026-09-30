//! Vocabulary loader and detokenizer, shared by every backend.
//!
//! Loads the JSON vocabulary file (token id → token string) and applies
//! detokenization rules ported from `parakeet-rs`'s `decoder_tdt.rs`:
//! special-token skip (any `<…>` token except `<unk>`), `▁` word-boundary
//! handling, and byte-fallback accumulation for `<0xNN>` pieces (checked
//! once at load — the real `parakeet_v3_vocab.json` has none, so this path
//! is exercised only by synthetic test vocabularies).

use std::collections::HashMap;
use std::path::Path;

use crate::token::TokenAt;
use crate::EngineError;

/// The two on-disk shapes of a vocabulary file — see [`Vocab::load`].
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum VocabFile {
    ById(Vec<String>),
    Keyed(HashMap<String, String>),
}

impl VocabFile {
    /// The id-indexed token table: a gap in a keyed file is `None`, and a
    /// key that is not a `u32` id is ignored.
    fn into_tokens(self) -> Vec<Option<String>> {
        match self {
            Self::ById(pieces) => pieces.into_iter().map(Some).collect(),
            Self::Keyed(map) => {
                let ids: Vec<(usize, String)> = map
                    .into_iter()
                    .filter_map(|(id, piece)| Some((id.parse::<u32>().ok()? as usize, piece)))
                    .collect();
                let len = ids.iter().map(|(id, _)| id + 1).max().unwrap_or(0);
                let mut tokens = vec![None; len];
                for (id, piece) in ids {
                    tokens[id] = Some(piece);
                }
                tokens
            }
        }
    }
}

/// Vocabulary: indexed by token id. `None` at an id means "no token with
/// this id in the JSON file" (a gap, or a backend's reserved id that the
/// file deliberately omits — see [`Vocab::load`]'s `reserved_id`).
#[derive(Debug, Clone)]
pub(crate) struct Vocab {
    /// Token string for each id.
    tokens: Vec<Option<String>>,
    /// Whether the vocab contains byte-fallback pieces like `<0x1A>`.
    has_byte_fallback: bool,
}

impl Vocab {
    /// Load the JSON vocabulary from `path`.
    ///
    /// Two shapes are accepted, both meaning "token id → token string":
    /// an object keyed by id as a string, e.g.
    /// `{"0":"<unk>","1":"<|nospeech|>","2":"▁hello",...}`, or an array
    /// whose index is the id, e.g. `["<unk>","<|nospeech|>","▁hello",...]`.
    ///
    /// `reserved_id` is an id the backend's decoder can emit but the
    /// vocabulary file omits (Parakeet-TDT's blank); the table is sized to
    /// cover it so [`Self::detokenize`] skips it instead of panicking. A
    /// backend whose file already covers every emittable id passes `None`
    /// — this is why the sizing rule is a parameter rather than a reach
    /// into one backend's constant.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if the file cannot be read or parsed.
    pub(crate) fn load(path: &Path, reserved_id: Option<u32>) -> Result<Self, EngineError> {
        let data = std::fs::read_to_string(path).map_err(|e| {
            EngineError::LoadFailed(format!("failed to read vocab {}: {e}", path.display()))
        })?;
        Self::from_json_str(&data, reserved_id)
    }

    /// Build the vocabulary from JSON text in either shape [`Self::load`]
    /// accepts, sized to cover `reserved_id`.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if `json` is neither shape.
    pub(crate) fn from_json_str(json: &str, reserved_id: Option<u32>) -> Result<Self, EngineError> {
        let file: VocabFile = serde_json::from_str(json)
            .map_err(|e| EngineError::LoadFailed(format!("failed to parse vocab JSON: {e}")))?;

        let mut tokens = file.into_tokens();
        let covered = reserved_id.map_or(0, |id| id as usize + 1);
        if tokens.len() < covered {
            tokens.resize(covered, None);
        }
        let has_byte_fallback = tokens.iter().flatten().any(|t| is_byte_fallback(t));

        Ok(Self {
            tokens,
            has_byte_fallback,
        })
    }

    /// Detokenize a sequence of tokens into text.
    ///
    /// Rules (ported from `parakeet-rs`'s `decoder_tdt.rs`):
    /// - Any token id with no vocabulary entry (including blank) is skipped.
    /// - Any token matching `<…>` is skipped **except** `<unk>`, which is
    ///   emitted literally.
    /// - A `▁`-prefixed token starts a new word: the marker is stripped and
    ///   a space is inserted before it, unless it is the very first output.
    /// - The real `parakeet_v3_vocab.json` doesn't actually use `▁` at all:
    ///   its word-initial pieces carry a literal leading space baked into
    ///   the token text itself (e.g. `" Ask"`), which already provides
    ///   correct inter-word spacing when appended as-is — except for the
    ///   very first piece emitted, whose leading space would otherwise leak
    ///   into the output as a bogus leading space on every single
    ///   transcript. That leading space is trimmed, but only at assembly
    ///   start (the first piece actually emitted, after special-token/byte-
    ///   fallback skipping) — every subsequent piece's embedded leading
    ///   space is legitimate inter-word spacing and is left untouched.
    /// - If the vocabulary has byte-fallback pieces, consecutive `<0xNN>`
    ///   tokens accumulate into a byte buffer that is flushed as UTF-8
    ///   whenever a non-byte-fallback token follows (or at the end).
    pub(crate) fn detokenize(&self, tokens: &[TokenAt]) -> String {
        let mut out = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();

        for token_at in tokens {
            let Some(token) = self
                .tokens
                .get(token_at.id as usize)
                .and_then(Option::as_deref)
            else {
                continue;
            };

            if self.has_byte_fallback && is_byte_fallback(token) {
                if let Some(byte) = parse_byte_fallback(token) {
                    byte_buf.push(byte);
                    continue;
                }
            }
            flush_byte_buf(&mut out, &mut byte_buf);

            if token.starts_with('<') && token.ends_with('>') && token != "<unk>" {
                continue; // special token — skip
            }

            if let Some(content) = token.strip_prefix('▁') {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(content);
            } else if out.is_empty() {
                // First piece actually emitted, real-vocab convention: trim
                // its one leading space (if any) instead of leaking it into
                // the output — see this method's doc comment.
                out.push_str(token.strip_prefix(' ').unwrap_or(token));
            } else {
                out.push_str(token);
            }
        }
        flush_byte_buf(&mut out, &mut byte_buf);

        out
    }

    /// `(is_word_initial, raw_piece_text)` for cross-window/cross-partial
    /// seam word segmentation — see `stream::merge::merge`'s `piece`
    /// parameter.
    ///
    /// A token is word-initial when its piece starts with the `▁`
    /// `SentencePiece` word-boundary marker, or (this vocab's actual
    /// convention — verified against `parakeet_v3_vocab.json`, which has
    /// no `▁` characters at all) a literal leading whitespace character
    /// (`" Amer"` + `"ic"` + `"ans"` → "Americans"). Everything else
    /// (subword continuations, and standalone punctuation like `","`)
    /// attaches to the word already open.
    ///
    /// Returns `None` for a token with no vocabulary entry (including
    /// blank).
    ///
    /// Returns the piece text borrowed from `self` (`&str`, not an owned
    /// `String` clone per call, WP9) — every real caller (`merge::merge`'s
    /// `piece` closure and its helpers) only reads the text transiently to
    /// build its own owned `Word::core`/`raw` `String`, never stores the
    /// borrow itself.
    pub(crate) fn piece_info(&self, id: u32) -> Option<(bool, &str)> {
        let token = self.tokens.get(id as usize)?.as_deref()?;
        let is_word_initial = token.starts_with('▁') || token.starts_with(char::is_whitespace);
        Some((is_word_initial, token))
    }
}

/// Whether `token` looks like a byte-fallback piece: `<0xNN>` (exactly two
/// hex digits — 6 bytes total: `<`, `0`, `x`, two hex digits, `>`).
fn is_byte_fallback(token: &str) -> bool {
    token.len() == 6 && token.starts_with("<0x") && token.ends_with('>')
}

/// Parse the hex byte out of a `<0xNN>` token.
fn parse_byte_fallback(token: &str) -> Option<u8> {
    u8::from_str_radix(&token[3..5], 16).ok()
}

/// Flush accumulated byte-fallback bytes as UTF-8 (lossy — a malformed
/// sequence still produces output rather than silently dropping text).
fn flush_byte_buf(out: &mut String, byte_buf: &mut Vec<u8>) {
    if byte_buf.is_empty() {
        return;
    }
    out.push_str(&String::from_utf8_lossy(byte_buf));
    byte_buf.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parakeet::tdt::BLANK;
    use crate::token::TokenAt;

    fn make_vocab(pieces: &[Option<&str>]) -> Vocab {
        let tokens: Vec<Option<String>> = pieces.iter().map(|p| p.map(str::to_string)).collect();
        let has_byte_fallback = tokens.iter().flatten().any(|t| is_byte_fallback(t));
        Vocab {
            tokens,
            has_byte_fallback,
        }
    }

    fn tok(id: u32) -> TokenAt {
        TokenAt { id, pos: 0 }
    }

    /// Round-trip: a hand-built vocab with ▁ boundaries, special tokens, and a multi-piece word.
    #[test]
    fn detokenize_handles_word_boundaries() {
        let vocab = make_vocab(&[
            Some("<unk>"),
            Some("<|nospeech|>"),
            Some("▁hello"),
            Some("▁world"),
            Some("!"),
        ]);
        let tokens = vec![tok(2), tok(3), tok(4)];
        assert_eq!(vocab.detokenize(&tokens), "hello world!");
    }

    /// Special tokens (`<|nospeech|>` etc, but not `<unk>`) are skipped.
    #[test]
    fn detokenize_skips_special_tokens_but_not_unk() {
        let vocab = make_vocab(&[Some("<unk>"), Some("<|nospeech|>"), Some("▁hello")]);
        let tokens = vec![tok(0), tok(1), tok(2)];
        // <unk> is rendered literally (per parakeet-rs), <|nospeech|> is skipped.
        assert_eq!(vocab.detokenize(&tokens), "<unk> hello");
    }

    /// Blank token (8192) has no vocabulary entry and is always skipped.
    #[test]
    fn detokenize_skips_blank() {
        let mut pieces = vec![None; 8193];
        pieces[0] = Some("▁hello");
        let vocab = make_vocab(&pieces);
        let tokens = vec![TokenAt { id: BLANK, pos: 0 }, tok(0)];
        assert_eq!(vocab.detokenize(&tokens), "hello");
    }

    /// Multi-piece word: "▁like" + "1" + "0" + "0" → "like100" (no
    /// digit-spacing heuristic here — that's a decode-time concern the
    /// upstream reference applies separately; our detokenizer only handles
    /// ▁ boundaries).
    #[test]
    fn detokenize_multi_piece_word() {
        let vocab = make_vocab(&[Some("▁like"), Some("1"), Some("0"), Some("0")]);
        let tokens = vec![tok(0), tok(1), tok(2), tok(3)];
        assert_eq!(vocab.detokenize(&tokens), "like100");
    }

    /// Leading ▁ on the first word should NOT add a leading space.
    #[test]
    fn detokenize_no_leading_space() {
        let vocab = make_vocab(&[Some("▁hello")]);
        let tokens = vec![tok(0)];
        assert_eq!(vocab.detokenize(&tokens), "hello");
    }

    /// D8 regression: the *real* `parakeet_v3_vocab.json` convention is a
    /// literal leading space baked into word-initial pieces (no `▁` at
    /// all) — the very first piece emitted must still not leak that space
    /// into the output, while every subsequent word-initial piece's
    /// embedded leading space is correct inter-word spacing and must be
    /// preserved untouched (interior spacing is not collapsed/altered).
    #[test]
    fn detokenize_trims_leading_space_on_first_piece_only_real_vocab_convention() {
        let vocab = make_vocab(&[Some(" Ask"), Some(" not"), Some(",")]);
        let tokens = vec![tok(0), tok(1), tok(2)];
        assert_eq!(vocab.detokenize(&tokens), "Ask not,");
    }

    /// The same first-piece trim must apply even when the special-token/
    /// byte-fallback skip logic means the leading-space piece isn't
    /// literally the first token in the slice — "first piece actually
    /// emitted" is what's trimmed, not "first token".
    #[test]
    fn detokenize_trims_leading_space_on_first_emitted_piece_after_skipped_special_token() {
        let vocab = make_vocab(&[Some("<|nospeech|>"), Some(" Hello")]);
        let tokens = vec![tok(0), tok(1)];
        assert_eq!(vocab.detokenize(&tokens), "Hello");
    }

    /// An id with no vocabulary entry at all (a gap) is skipped, not a panic.
    #[test]
    fn detokenize_skips_unknown_id() {
        let vocab = make_vocab(&[Some("▁hi")]);
        let tokens = vec![TokenAt { id: 99, pos: 0 }, tok(0)];
        assert_eq!(vocab.detokenize(&tokens), "hi");
    }

    /// Byte-fallback pieces accumulate into a multibyte UTF-8 character.
    /// "é" is U+00E9, UTF-8 bytes `0xC3 0xA9`.
    #[test]
    fn detokenize_byte_fallback_multibyte_utf8() {
        let vocab = make_vocab(&[Some("<0xC3>"), Some("<0xA9>"), Some("▁word")]);
        let tokens = vec![tok(0), tok(1), tok(2)];
        assert_eq!(vocab.detokenize(&tokens), "é word");
    }

    /// A vocab without byte-fallback pieces never treats `<0xNN>`-shaped
    /// tokens specially (the flag is per-vocab, decided once at load).
    #[test]
    fn detokenize_no_byte_fallback_when_absent_from_vocab() {
        let vocab = Vocab {
            tokens: vec![Some("<0xC3>".to_string()), Some("▁word".to_string())],
            has_byte_fallback: false,
        };
        let tokens = vec![tok(0), tok(1)];
        // Without the byte-fallback flag, "<0xC3>" is just another `<…>`
        // special token and gets skipped.
        assert_eq!(vocab.detokenize(&tokens), "word");
    }

    /// `piece_info` recognizes both word-boundary conventions: the
    /// `SentencePiece` `▁` marker and a literal leading space (this vocab's
    /// actual convention).
    #[test]
    fn piece_info_recognizes_word_boundary_markers() {
        let vocab = make_vocab(&[Some("▁Not"), Some(" ask"), Some("ic")]);
        assert_eq!(vocab.piece_info(0), Some((true, "▁Not")));
        assert_eq!(vocab.piece_info(1), Some((true, " ask")));
        assert_eq!(
            vocab.piece_info(2),
            Some((false, "ic")),
            "a continuation piece is not word-initial"
        );
    }

    /// Blank / unknown ids have no vocabulary entry — `piece_info` is `None`.
    #[test]
    fn piece_info_none_for_missing_entry() {
        let vocab = make_vocab(&[Some("▁hi")]);
        assert_eq!(vocab.piece_info(BLANK), None);
        assert_eq!(vocab.piece_info(99), None);
    }

    fn vocab_from(json: &str, reserved_id: Option<u32>) -> Vocab {
        Vocab::from_json_str(json, reserved_id).expect("valid vocab fixture")
    }

    /// The id-keyed object shape loads with gaps as `None`, sized to cover
    /// the reserved id.
    #[test]
    fn load_reads_an_id_keyed_object() {
        let vocab = vocab_from(r#"{"0":"<unk>","2":"▁hi"}"#, Some(4));
        assert_eq!(vocab.piece_info(2), Some((true, "▁hi")));
        assert_eq!(vocab.piece_info(1), None, "a gap has no entry");
        assert_eq!(vocab.piece_info(4), None, "the reserved id has no entry");
        assert_eq!(vocab.tokens.len(), 5);
    }

    /// The array shape uses the index as the id — the same table the
    /// object shape would give.
    #[test]
    fn load_reads_an_id_indexed_array() {
        let vocab = vocab_from(r#"["<unk>","▁hi","ic"]"#, Some(3));
        assert_eq!(vocab.piece_info(1), Some((true, "▁hi")));
        assert_eq!(vocab.piece_info(2), Some((false, "ic")));
        assert_eq!(vocab.piece_info(3), None, "the reserved id has no entry");
        assert_eq!(vocab.tokens.len(), 4);
        assert_eq!(vocab.detokenize(&[tok(1), tok(2), tok(3)]), "hiic");
    }

    #[test]
    fn load_rejects_a_file_that_is_neither_shape() {
        let err = Vocab::from_json_str("42", None).expect_err("a bare number is not a vocabulary");
        assert!(matches!(err, EngineError::LoadFailed(_)));
    }
}
