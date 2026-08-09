//! Canary's decoder prompt: the fixed scaffold plus the source/target
//! language tokens that select transcription.
//!
//! Canary is an attention encoder-decoder that must be *told* what language
//! it is hearing — there is no auto-detect. Vuho only ever transcribes
//! (source == target); translation would need its own target-language UI
//! and is out of scope.

/// Number of prompt tokens the decoder is seeded with before free decoding
/// starts.
pub const PROMPT_LEN: usize = 10;

/// Index of the source-language slot in [`SCAFFOLD`].
const SRC_LANG_SLOT: usize = 4;
/// Index of the target-language slot in [`SCAFFOLD`].
const TGT_LANG_SLOT: usize = 5;

/// The `canary2` prompt layout, with the two language slots left at a
/// placeholder that [`transcribe_prompt`] fills.
///
/// Verified against the shipped `vocab.json` (see
/// `tests/canary_batch.rs`, which asserts every one of these ids against
/// the file rather than against this table):
/// `▁`, `<|startofcontext|>`, `<|startoftranscript|>`, `<|emo:undefined|>`,
/// `<src>`, `<tgt>`, `<|pnc|>`, `<|noitn|>`, `<|notimestamp|>`,
/// `<|nodiarize|>`.
const SCAFFOLD: [i32; PROMPT_LEN] = [16053, 7, 4, 16, 0, 0, 5, 9, 11, 13];

/// The language the load-time warmup inference decodes silence as. Its only
/// requirement is membership in [`LANGUAGES`], pinned by a unit test below.
pub(crate) const WARMUP_LANGUAGE: &str = "en";

/// End-of-sequence id: the decode loop stops here and does not emit it.
pub const EOS_ID: u32 = 3;

/// The 25 languages Canary-1B-v2 supports, each with its `<|xx|>`
/// vocabulary id. The single source of truth for both — a second copy of
/// the code list anywhere else would be a list to keep in sync (rule 26).
const LANGUAGES: [(&str, i32); 25] = [
    ("bg", 46),
    ("cs", 59),
    ("da", 60),
    ("de", 78),
    ("el", 79),
    ("en", 64),
    ("es", 171),
    ("et", 66),
    ("fi", 70),
    ("fr", 71),
    ("hr", 58),
    ("hu", 89),
    ("it", 99),
    ("lt", 120),
    ("lv", 117),
    ("mt", 127),
    ("nl", 62),
    ("pl", 150),
    ("pt", 151),
    ("ro", 154),
    ("ru", 157),
    ("sk", 167),
    ("sl", 168),
    ("sv", 175),
    ("uk", 192),
];

/// The `<|xx|>` token id for a language code, or `None` if Canary cannot
/// transcribe that language.
#[must_use]
pub fn lang_token(code: &str) -> Option<i32> {
    LANGUAGES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|&(_, id)| id)
}

/// The transcribe prompt for `language` — the same language id in both the
/// source and target slots, which is what makes this transcription rather
/// than translation.
#[must_use]
pub fn transcribe_prompt(language: &str) -> Option<[i32; PROMPT_LEN]> {
    let id = lang_token(language)?;
    let mut prompt = SCAFFOLD;
    prompt[SRC_LANG_SLOT] = id;
    prompt[TGT_LANG_SLOT] = id;
    Some(prompt)
}

/// Every language code this backend can transcribe.
pub fn supported_languages() -> impl Iterator<Item = &'static str> {
    LANGUAGES.iter().map(|&(code, _)| code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_transcribe_prompt_carries_the_same_language_in_both_slots() {
        let prompt = transcribe_prompt("en").expect("en is supported");
        assert_eq!(prompt[SRC_LANG_SLOT], prompt[TGT_LANG_SLOT]);
        assert_eq!(prompt[SRC_LANG_SLOT], lang_token("en").unwrap());
        assert_eq!(prompt.len(), PROMPT_LEN);
    }

    /// An unsupported language must produce no prompt at all, so the caller
    /// is forced to surface an error rather than silently transcribing as
    /// some other language (CONSTITUTION rule 2).
    #[test]
    fn an_unsupported_language_has_no_prompt() {
        assert_eq!(transcribe_prompt("ja"), None);
        assert_eq!(lang_token("ja"), None);
    }

    #[test]
    fn the_warmup_language_is_supported() {
        assert!(transcribe_prompt(WARMUP_LANGUAGE).is_some());
    }

    #[test]
    fn every_supported_language_has_a_prompt() {
        for code in supported_languages() {
            assert!(
                transcribe_prompt(code).is_some(),
                "{code} is listed as supported but has no prompt"
            );
        }
    }

    /// Two languages sharing a token id would silently transcribe as each
    /// other — the table's ids must be distinct, as must its codes.
    #[test]
    fn language_codes_and_ids_are_both_unique() {
        let mut ids: Vec<i32> = LANGUAGES.iter().map(|&(_, id)| id).collect();
        ids.sort_unstable();
        let unique = ids.len();
        ids.dedup();
        assert_eq!(
            ids.len(),
            unique,
            "duplicate language token id in the table"
        );

        let mut codes: Vec<&str> = supported_languages().collect();
        codes.sort_unstable();
        let unique = codes.len();
        codes.dedup();
        assert_eq!(codes.len(), unique, "duplicate language code in the table");
    }
}
