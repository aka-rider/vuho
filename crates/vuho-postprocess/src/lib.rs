//! Text post-processing for dictation output.
//!
//! Rule-based cleanup: filler removal, spacing normalization,
//! newline cleanup. No LLMs.

/// Result of post-processing.
#[derive(Clone, Debug)]
pub struct PostprocessedText {
    /// Cleaned text ready for injection.
    pub text: String,
}

/// Language-aware filler word dictionaries.
const FILLER_WORDS: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "uhm", "uh", "um", "erm", "er", "like", "you know", "so like",
        ],
    ),
    (
        "de",
        &["ähm", "äh", "hm", "na ja", "also", "wie soll ich sagen"],
    ),
    (
        "fr",
        &["euh", "ben", "alors", "donc", "en fait", "vous savez"],
    ),
    ("vi", &["ừm", "ờ", "uhm", "thì", " kiểu như"]),
];

/// Post-process raw transcription text.
///
/// Applies filler removal, spacing normalization, and newline cleanup.
/// `language` drives only the filler-word dictionary lookup
/// (`get_fillers`): an unrecognized code — including
/// `vuho-stt-engine`'s `"und"` (undetermined) sentinel for a language that
/// was never actually detected — simply finds no dictionary and skips
/// filler removal entirely (CONSTITUTION rule 24: conservative cleanup,
/// never guess at a language's disfluencies). Newline collapsing and
/// space/punctuation normalization are language-agnostic formatting (no
/// words are removed) and always run regardless of `language`.
#[must_use]
pub fn postprocess(text: &str, language: &str) -> PostprocessedText {
    let mut result = text.to_string();

    // Remove filler words (case-insensitive, whole-word match)
    if let Some(fillers) = get_fillers(language) {
        for filler in fillers {
            result = remove_fillers(&result, filler);
        }
    }

    // Normalize newlines: collapse multiple newlines into one
    result = collapse_newlines(&result);

    // Normalize spacing
    result = normalize_spaces(&result);

    // Trim and capitalize first letter if sentence-like
    result = result.trim().to_string();

    PostprocessedText { text: result }
}

/// Get filler words for a language.
fn get_fillers(language: &str) -> Option<&[&str]> {
    FILLER_WORDS
        .iter()
        .find(|(lang, _)| *lang == language)
        .map(|(_, words)| *words)
}

/// Remove filler words (case-insensitive, whole-word).
///
/// Operates entirely on `char` indices so that multi-byte UTF-8 characters
/// are handled correctly — no byte / char index confusion.
fn remove_fillers(text: &str, filler: &str) -> String {
    let filler_chars: Vec<char> = filler.to_lowercase().chars().collect();
    let filler_len = filler_chars.len();
    let text_chars: Vec<char> = text.chars().collect();

    let mut result = String::with_capacity(text.len());
    let mut i = 0;

    while i < text_chars.len() {
        // Check if filler starts at position i (char-by-char comparison).
        if i + filler_len <= text_chars.len() {
            let matches = (0..filler_len)
                .all(|j| text_chars[i + j].to_lowercase().next() == Some(filler_chars[j]));

            if matches {
                // Check word boundaries.
                let before_ok = i == 0 || !text_chars[i - 1].is_alphabetic();
                let after_pos = i + filler_len;
                let after_ok =
                    after_pos >= text_chars.len() || !text_chars[after_pos].is_alphabetic();

                if before_ok && after_ok {
                    // Skip the filler word and any trailing space.
                    i = after_pos;
                    if i < text_chars.len() && text_chars[i] == ' ' {
                        i += 1;
                    }
                    continue;
                }
            }
        }

        result.push(text_chars[i]);
        i += 1;
    }

    result
}

/// Collapse multiple consecutive newlines into double newline (paragraph break).
fn collapse_newlines(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut consecutive_newlines = 0;

    for ch in text.chars() {
        if ch == '\n' {
            consecutive_newlines += 1;
            if consecutive_newlines <= 2 {
                result.push(ch);
            }
        } else {
            consecutive_newlines = 0;
            result.push(ch);
        }
    }

    result
}

/// Normalize spacing: collapse multiple spaces, fix space before punctuation.
///
/// Handles ASCII punctuation (`. , ! ? : ;`) and common Unicode punctuation
/// used in CJK / Vietnamese (`。 ， ！ ？ ： ； 「 」 『 』 — ‥`).
fn normalize_spaces(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut pending_space = false;

    for ch in text.chars() {
        if ch == ' ' {
            if !pending_space {
                pending_space = true;
            }
        } else if pending_space {
            if !is_punctuation(ch) {
                // Space before non-punctuation: keep the space, then push the char.
                result.push(' ');
            }
            // Drop the pending space (either consumed by punctuation, or already pushed).
            pending_space = false;
            result.push(ch);
        } else {
            result.push(ch);
        }
    }

    result
}

/// Returns true if `ch` is a punctuation character that should not be
/// preceded by a space.
fn is_punctuation(ch: char) -> bool {
    // ASCII punctuation
    if matches!(ch, '.' | ',' | '!' | '?' | ':' | ';') {
        return true;
    }
    // Common Unicode punctuation: CJK / Vietnamese / general
    let code = ch as u32;
    matches!(code,
        // CJK punctuation (Ideographic punctuation & symbols)
        0x3001..=0x3003 | // 、 。 「
        0x3008..=0x3011 | // 『』【】《》
        0x3014..=0x3017 | // 〔〕〖〗
        0x301D..=0x302F | // double-angle 「」 + other CJK symbols
        // General punctuation (— … ‥ ‧ etc.)
        0x2010..=0x201F | // - ‐ ‑ ‒ — ‗ … ․ ‧
        0x2030..=0x205F | // ‰ ‱ … ‧ ∷ ∶ ∷
        // Fullwidth ASCII variants (！ ＂ ＃ … ～)
        0xFF01..=0xFF60,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_english_fillers() {
        let result = postprocess("Um, hello world, uh, how are you?", "en");
        assert!(!result.text.to_lowercase().contains("um,"));
        assert!(!result.text.to_lowercase().contains(" uh,"));
        assert!(result.text.contains("hello world"));
    }

    #[test]
    fn collapses_multiple_newlines() {
        let input = "Line one\n\n\n\nLine two\n\n\nLine three";
        let result = postprocess(input, "en");
        assert!(!result.text.contains("\n\n\n"));
    }

    #[test]
    fn normalizes_spaces() {
        let input = "Hello   world.  How are you?  Fine.";
        let result = postprocess(input, "en");
        assert!(!result.text.contains("  "));
        assert!(result.text.contains(". How"));
    }

    #[test]
    fn handles_empty_input() {
        let result = postprocess("", "en");
        assert_eq!(result.text, "");
    }

    /// Finding 7: an unrecognized language code (including the
    /// `vuho-stt-engine` "und" sentinel for "never actually detected") must
    /// not apply another language's filler rules, but formatting
    /// normalization still runs — it's language-agnostic and removes no
    /// words.
    #[test]
    fn unknown_language_skips_filler_removal_but_still_normalizes_formatting() {
        let result = postprocess("um  hello   world", "und");
        assert!(
            result.text.contains("um"),
            "an unrecognized language code must not apply English filler rules, got: {}",
            result.text
        );
        assert!(
            !result.text.contains("  "),
            "formatting normalization must still run regardless of language, got: {}",
            result.text
        );
    }

    #[test]
    fn removes_filler_after_non_ascii_char() {
        // "à" in "chào" is 2 UTF-8 bytes. "uhm" starts at char index 9
        // but byte index 10. The old implementation used char index
        // to index .as_bytes(), so it compared the wrong bytes.
        let result = postprocess("Xin chào uhm thế giới", "vi");
        assert!(
            !result.text.contains("uhm"),
            "filler 'uhm' after non-ASCII char should be removed, got: {}",
            result.text
        );
        assert!(
            result.text.contains("chào"),
            "original text should be preserved: {}",
            result.text
        );
    }

    #[test]
    fn normalizes_unicode_punctuation() {
        // ASCII
        let result = postprocess("Hello . World ,", "en");
        assert_eq!(result.text, "Hello. World,");

        // CJK / Vietnamese: ideographic period
        let result = postprocess("Xin chào 。", "vi");
        assert_eq!(result.text, "Xin chào。");

        // Em-dash: space before punctuation is removed
        let result = postprocess("Hello — world", "en");
        assert_eq!(result.text, "Hello— world");

        // Multiple spaces + Unicode punctuation
        let result = postprocess("Hello   。  World ，", "vi");
        assert_eq!(result.text, "Hello。 World，");
    }
}
