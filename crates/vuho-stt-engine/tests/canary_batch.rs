//! Canary-1B-v2 batch regression, driven through the public
//! `TranscriptionEngine` surface.
//!
//! Model-gated: every test skips (with an `eprintln`, not a failure) when
//! the Canary model is not provisioned, so CI without `models/` stays green
//! — the same pattern as `tests/batch_multiwindow.rs`.
//!
//! The model id is found by *backend*, never by name, so no model id is
//! written down outside `models.manifest.json` (ADR-019).

use std::collections::HashMap;
use std::path::PathBuf;

use vuho_stt_engine::bench_support::WINDOW_SAMPLES;
use vuho_stt_engine::canary::prompt;
use vuho_stt_engine::test_support::{jfk_wav_path, load_wav_16k_mono_f32};
use vuho_stt_engine::{CanaryEngine, EngineError, TranscriptionEngine};

/// The quote jfk.wav contains.
const EXPECTED_QUOTE: &str = "ask not what your country can do for you";

/// The manifest's Canary model id plus its resolved folder, or `None` when
/// this environment has no Canary model provisioned.
fn canary_model() -> Option<(&'static str, PathBuf)> {
    let id = vuho_stt_engine::canary::manifest_model_id().or_else(|| {
        eprintln!("skipping: the manifest declares no Canary model");
        None
    })?;
    match vuho_stt_engine::resolve_model_folder(id) {
        Ok(folder) => Some((id, folder)),
        Err(e) => {
            eprintln!("skipping: no Canary model folder resolved: {e}");
            None
        }
    }
}

fn load_engine() -> Option<CanaryEngine> {
    let (id, folder) = canary_model()?;
    Some(CanaryEngine::load(id, folder).expect("Canary engine load"))
}

fn load_jfk() -> Option<Vec<f32>> {
    let path = jfk_wav_path().or_else(|| {
        eprintln!("skipping: JFK_WAV/jfk.wav not found in this environment");
        None
    })?;
    Some(load_wav_16k_mono_f32(&path).expect("parse jfk.wav"))
}

/// (a) Every Canary constant this crate hardcodes, checked against the
/// **shipped model files** rather than against itself.
///
/// A wrong prompt token id yields fluent-but-wrong output with an otherwise
/// green suite, so a table-vs-itself assertion would be worthless: the
/// language ids, the ten prompt slots, and eos/bos/pad are all read out of
/// `vocab.json` and `metadata.json` here.
#[test]
fn canary_constants_agree_with_the_shipped_model_files() {
    let Some((id, folder)) = canary_model() else {
        return;
    };
    let model = vuho_model_paths::manifest()
        .stt
        .model(id)
        .expect("the manifest model we were given the id of");

    let vocab_file = folder.join(model.asset("vocab").expect("a vocab asset"));
    let vocab: HashMap<String, String> =
        serde_json::from_str(&std::fs::read_to_string(&vocab_file).expect("read vocab.json"))
            .expect("parse vocab.json");
    let piece = |id: i32| -> &str {
        vocab
            .get(&id.to_string())
            .unwrap_or_else(|| panic!("vocab.json has no entry for id {id}"))
    };

    for code in prompt::supported_languages() {
        let token = prompt::lang_token(code).expect("a listed language has a token");
        assert_eq!(
            piece(token),
            format!("<|{code}|>"),
            "language token {token} for {code} does not name that language in vocab.json"
        );
    }

    // The prompt's non-language slots are the fixed scaffold; the two
    // language slots must both carry the requested language, which is what
    // makes this transcription rather than translation.
    let expected_scaffold = [
        (0, "▁"),
        (1, "<|startofcontext|>"),
        (2, "<|startoftranscript|>"),
        (3, "<|emo:undefined|>"),
        (4, "<|en|>"),
        (5, "<|en|>"),
        (6, "<|pnc|>"),
        (7, "<|noitn|>"),
        (8, "<|notimestamp|>"),
        (9, "<|nodiarize|>"),
    ];
    let built = prompt::transcribe_prompt("en").expect("en is supported");
    assert_eq!(built.len(), prompt::PROMPT_LEN);
    for (slot, expected) in expected_scaffold {
        assert_eq!(
            piece(built[slot]),
            expected,
            "prompt slot {slot} (id {}) is not {expected}",
            built[slot]
        );
    }

    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(folder.join("metadata.json")).expect("read"))
            .expect("parse metadata.json");
    let meta_id = |key: &str| -> i32 {
        i32::try_from(metadata[key].as_i64().unwrap_or_else(|| panic!("no {key}")))
            .expect("a small id")
    };
    assert_eq!(
        u32::try_from(meta_id("eos_id")).expect("a non-negative id"),
        prompt::EOS_ID,
        "the decode loop's EOS id disagrees with the shipped metadata"
    );
    assert_eq!(piece(meta_id("eos_id")), "<|endoftext|>");
    assert_eq!(piece(meta_id("pad_id")), "<pad>");
    assert_eq!(
        meta_id("bos_id"),
        built[2],
        "bos must be the <|startoftranscript|> slot the prompt seeds"
    );
}

/// (b) The whole point: a real 11 s utterance transcribes correctly.
///
/// (f) also logs the wall clock of one single-window `transcribe`, which is
/// both the batch cost and — since a stop runs exactly one end-aligned
/// final-window inference — the dominant term in the stop→text latency.
#[test]
fn canary_transcribes_jfk_wav() {
    let Some(engine) = load_engine() else { return };
    let Some(samples) = load_jfk() else { return };

    let started = std::time::Instant::now();
    let result = engine.transcribe(&samples, Some("en")).expect("transcribe");
    let elapsed = started.elapsed();
    log::info!("canary: single-window transcribe took {elapsed:?}");
    println!("canary: single-window transcribe took {elapsed:?}");

    let lower = result.full_text.to_lowercase();
    assert!(
        lower.contains(EXPECTED_QUOTE),
        "expected the JFK quote, got: {}",
        result.full_text
    );
}

/// (c) The falsifiable check for the fixed-stride positions (A2): a buffer
/// longer than one 15 s window must cross the seam without dropping or
/// duplicating content, using text-only overlap matching.
#[test]
fn canary_crosses_a_window_seam_without_dropping_or_duplicating() {
    let Some(engine) = load_engine() else { return };
    let Some(one) = load_jfk() else { return };

    let mut samples = Vec::with_capacity(one.len() * 3);
    for _ in 0..3 {
        samples.extend_from_slice(&one);
    }
    assert!(
        samples.len() > WINDOW_SAMPLES,
        "the concatenation must exceed one window to exercise a seam"
    );

    let started = std::time::Instant::now();
    let result = engine.transcribe(&samples, Some("en")).expect("transcribe");
    println!(
        "canary: {} samples of audio transcribed in {:?}",
        samples.len(),
        started.elapsed()
    );

    let lower = result.full_text.to_lowercase();
    assert_eq!(
        lower.matches(EXPECTED_QUOTE).count(),
        3,
        "expected the quote exactly 3 times (once per repetition, no seam dup/drop), got: {}",
        result.full_text
    );
    assert_eq!(
        lower
            .matches("ask what you can do for your country")
            .count(),
        3,
        "expected the second half of the quote exactly 3 times, got: {}",
        result.full_text
    );
}

/// (d) A language Canary cannot transcribe is a typed error, never a
/// silent fallback to some other language (CONSTITUTION rule 2).
#[test]
fn canary_rejects_an_unsupported_language() {
    let Some(engine) = load_engine() else { return };

    let err = engine
        .transcribe(&vec![0.0f32; 16_000], Some("ja"))
        .expect_err("ja is outside Canary's 25 languages");
    match err {
        EngineError::UnsupportedLanguage { model, language } => {
            assert_eq!(language, "ja");
            assert!(!model.is_empty(), "the error must name the model");
        }
        other => panic!("expected UnsupportedLanguage, got {other:?}"),
    }
}

/// (e) Canary's specials are pipe-delimited (`<|en|>`); the shared
/// detokenizer must not leak any of them into user-visible text (A3).
#[test]
fn canary_detokenization_leaks_no_special_tokens() {
    let Some(engine) = load_engine() else { return };
    let Some(samples) = load_jfk() else { return };

    let result = engine.transcribe(&samples, Some("en")).expect("transcribe");
    assert!(
        !result.full_text.contains("<|"),
        "a special token leaked into the transcript: {}",
        result.full_text
    );
    for segment in &result.segments {
        assert!(
            !segment.text.contains("<|"),
            "a special token leaked into a segment: {}",
            segment.text
        );
    }
}
