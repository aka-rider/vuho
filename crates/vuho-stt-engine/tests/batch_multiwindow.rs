//! Multi-window batch regression: jfk.wav repeated 3× (≈33 s, spanning three
//! 15 s model windows with 2 s overlap) must transcribe the JFK quote three
//! times with **no duplicated or dropped words at the window seams**.
//!
//! Model-gated: skips (with an `eprintln`, not a failure) when no model is
//! provisioned, so CI without `models/` stays green — same pattern as the
//! other model-gated tests in this crate (`coreml.rs`, `vad.rs`).
//!
//! WAV loading is `vuho_stt_engine::test_support`'s job (behind the
//! `test-fixtures` feature, see this crate's `Cargo.toml`) — the one place
//! in the workspace that logic lives (CONSTITUTION rule 26), also used by
//! `test-stt-ffi` and this crate's own model-gated unit tests.

use vuho_stt_engine::test_support::{jfk_wav_path, load_wav_16k_mono_f32};
use vuho_stt_engine::{ParakeetEngine, TranscriptionEngine};

/// jfk.wav repeated 3×, with no dedup/duplication regressions at the two
/// window seams this produces (`windower::plan` puts seams inside the sliding
/// window's overlap region, exactly where `merge.rs`'s dedup logic runs).
///
/// **Previously `#[ignore]`d** with a real, characterized cross-window
/// content-drop bug (not a merge/dedup bug): with three back-to-back
/// repetitions, the middle repetition's tail used to be silently dropped
/// entirely — window 1, primed with decoder state carried over from
/// window 0's *last* emission (mid-sentence, "…ask not"), never crossed
/// the blank threshold for its entire 15 s span, even though the correct
/// next token was its runner-up by a margin as small as ~0.5 in log-space
/// at the exact frame where it should have resynced. The content was never
/// decoded by any window, so no merge/dedup logic could recover it.
///
/// Fixed by porting `FluidAudio`'s actual `ChunkProcessor.swift` design
/// (`engine.rs`'s `transcribe`, `stream::merge`): each window now decodes
/// from a **fresh** decoder state instead of one carried across windows —
/// the root cause of the permanent blank-lock — and `stream::merge`
/// reconciles the resulting overlap at **word** granularity (not raw token
/// id), tolerant of the seam-only case/subword-split/punctuation
/// disagreement two independent decodes of the same audio produce. See the
/// design notes in `engine.rs` and `stream/merge.rs` for the full
/// rationale.
#[test]
fn jfk_repeated_three_times_has_no_seam_duplication() {
    let Some(path) = jfk_wav_path() else {
        eprintln!("skipping: JFK_WAV/jfk.wav not found in this environment");
        return;
    };
    let Ok(model_folder) = vuho_stt_engine::resolve_model_folder() else {
        eprintln!("skipping: no model folder resolved in this environment");
        return;
    };

    let one = load_wav_16k_mono_f32(&path).expect("parse jfk.wav");
    let mut samples = Vec::with_capacity(one.len() * 3);
    samples.extend_from_slice(&one);
    samples.extend_from_slice(&one);
    samples.extend_from_slice(&one);

    let engine = ParakeetEngine::load(model_folder).expect("engine load");
    let result = engine.transcribe(&samples, Some("en")).expect("transcribe");

    let lower = result.full_text.to_lowercase();
    let occurrences = lower
        .matches("ask not what your country can do for you")
        .count();
    assert_eq!(
        occurrences, 3,
        "expected the quote exactly 3 times (once per repetition, no seam dup/drop), got {occurrences}: {}",
        result.full_text
    );

    // The window-seam merge must not duplicate a word run: if it did, the
    // most likely artifact is the SAME sentence appearing 4+ times (an
    // extra partial repeat from a seam that failed to dedup) or fewer than
    // 3 (a seam that over-dropped and swallowed a repetition).
    let ask_country_occurrences = lower
        .matches("ask what you can do for your country")
        .count();
    assert_eq!(
        ask_country_occurrences, 3,
        "expected the second half of the quote exactly 3 times, got {ask_country_occurrences}: {}",
        result.full_text
    );
}
