//! Voz batch regression, driven through the public `TranscriptionEngine`
//! surface.
//!
//! Model-gated: every test skips (with an `eprintln`, not a failure) when
//! the voz model is not provisioned, so CI without `models/` stays green —
//! the same pattern as `tests/canary_batch.rs`.
//!
//! The model id is found by *backend*, never by name, so no model id is
//! written down outside `models.manifest.json` (ADR-019).

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::time::Instant;

use vuho_stt_engine::bench_support::WINDOW_SAMPLES;
use vuho_stt_engine::test_support::{jfk_wav_path, load_wav_16k_mono_f32};
use vuho_stt_engine::{TranscriptionEngine, VozEngine};

/// The quote jfk.wav contains.
const EXPECTED_QUOTE: &str = "ask not what your country can do for you";

/// The manifest's voz model id plus its resolved folder, or `None` when
/// this environment has no voz model provisioned.
fn voz_model() -> Option<(&'static str, PathBuf)> {
    let id = vuho_stt_engine::voz::manifest_model_id().or_else(|| {
        eprintln!("skipping: the manifest declares no voz model");
        None
    })?;
    match vuho_stt_engine::resolve_model_folder(id) {
        Ok(folder) => Some((id, folder)),
        Err(e) => {
            eprintln!("skipping: no voz model folder resolved: {e}");
            None
        }
    }
}

fn load_engine() -> Option<VozEngine> {
    let (id, folder) = voz_model()?;
    let started = Instant::now();
    let engine = VozEngine::load(id, folder).expect("voz engine load");
    println!("voz: VozEngine::load took {:?}", started.elapsed());
    Some(engine)
}

fn load_jfk() -> Option<Vec<f32>> {
    let path = jfk_wav_path().or_else(|| {
        eprintln!("skipping: JFK_WAV/jfk.wav not found in this environment");
        None
    })?;
    Some(load_wav_16k_mono_f32(&path).expect("parse jfk.wav"))
}

/// (a) Every number `meta.json` carries, checked against the **shipped
/// model files** rather than against this crate.
///
/// `VozEngine::load` already refuses a `meta.json` whose tensor shapes
/// disagree with the models' own (a mismatch is `LoadFailed`), so a
/// successful load is the shape check; what is left to assert here is the
/// vocabulary file, which `meta.json` describes but the models do not.
#[test]
fn voz_constants_agree_with_the_shipped_model_files() {
    let Some((id, folder)) = voz_model() else {
        return;
    };
    let model = vuho_model_paths::manifest()
        .stt
        .model(id)
        .expect("the manifest model we were given the id of");
    let read_json = |role: &str| -> serde_json::Value {
        let file = folder.join(
            model
                .asset(role)
                .unwrap_or_else(|| panic!("a {role} asset")),
        );
        serde_json::from_str(&std::fs::read_to_string(&file).expect("read")).expect("parse")
    };

    let meta = read_json("meta");
    let number = |key: &str| -> u64 {
        meta[key]
            .as_u64()
            .unwrap_or_else(|| panic!("meta.json has no numeric {key}"))
    };
    let vocab = read_json("vocab");
    let pieces = vocab.as_array().expect("vocab.json is an array of pieces");

    assert_eq!(
        u64::try_from(pieces.len()).expect("a small count"),
        number("vocab_size"),
        "vocab.json must hold exactly vocab_size pieces"
    );
    assert_eq!(
        number("blank_idx"),
        number("vocab_size"),
        "the blank is the id just past the vocabulary"
    );
    assert_eq!(
        number("n_logits"),
        number("vocab_size") + 1 + number("num_durations")
    );

    load_engine().expect("a load that validates meta.json against the models' shapes");
}

/// (b) The whole point: a real 11 s utterance transcribes correctly.
///
/// Also prints the wall clock of one single-window `transcribe`, which is
/// both the batch cost and — since a stop runs exactly one end-aligned
/// final-window inference — the dominant term in the stop→text latency.
#[test]
fn voz_transcribes_jfk_wav() {
    let Some(engine) = load_engine() else { return };
    let Some(samples) = load_jfk() else { return };

    let started = Instant::now();
    let result = engine.transcribe(&samples, Some("en")).expect("transcribe");
    println!("voz: single-window transcribe took {:?}", started.elapsed());

    let lower = result.full_text.to_lowercase();
    assert!(
        lower.contains(EXPECTED_QUOTE),
        "expected the JFK quote, got: {}",
        result.full_text
    );
}

/// (c) A buffer longer than one 15 s window must cross the seam without
/// dropping or duplicating content — voz's positions are measured encoder
/// frames, so the shared merge works on them as it does for Parakeet.
#[test]
fn voz_crosses_a_window_seam_without_dropping_or_duplicating() {
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

    let started = Instant::now();
    let result = engine.transcribe(&samples, Some("en")).expect("transcribe");
    println!(
        "voz: {} samples of audio transcribed in {:?}",
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

/// (d) Voz's vocabulary carries `<…>` specials; the shared detokenizer
/// must not leak any of them into user-visible text.
#[test]
fn voz_detokenization_leaks_no_special_tokens() {
    let Some(engine) = load_engine() else { return };
    let Some(samples) = load_jfk() else { return };

    let result = engine.transcribe(&samples, Some("en")).expect("transcribe");
    let leaks_a_special = |text: &str| {
        text.find('<')
            .is_some_and(|open| text[open..].contains('>'))
    };
    assert!(
        !leaks_a_special(&result.full_text),
        "a special token leaked into the transcript: {}",
        result.full_text
    );
    for segment in &result.segments {
        assert!(
            !leaks_a_special(&segment.text),
            "a special token leaked into a segment: {}",
            segment.text
        );
    }
}
