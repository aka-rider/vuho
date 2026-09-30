//! ONNX Parakeet batch regression (Linux): jfk.wav transcribes to the quote,
//! and jfk.wav repeated 3× (≈33 s, crossing the 15 s window seams) contains
//! each half of the quote exactly three times — nothing dropped or
//! duplicated at a seam.
//!
//! Model-gated: skips (with an `eprintln`, not a failure) when the model is
//! not provisioned, like `tests/batch_multiwindow.rs` does on macOS.

#![cfg(target_os = "linux")]

use std::time::Instant;

use vuho_stt_engine::test_support::{jfk_wav_path, load_wav_16k_mono_f32};
use vuho_stt_engine::{OnnxParakeetEngine, TranscriptionEngine};

const FIRST_HALF: &str = "ask not what your country can do for you";
const SECOND_HALF: &str = "ask what you can do for your country";

fn load_engine_and_jfk() -> Option<(OnnxParakeetEngine, Vec<f32>)> {
    let Some(path) = jfk_wav_path() else {
        eprintln!("skipping: JFK_WAV/jfk.wav not found in this environment");
        return None;
    };
    let model_id = vuho_model_paths::manifest().stt.default_model();
    let Ok(folder) = vuho_stt_engine::resolve_model_folder(model_id) else {
        eprintln!("skipping: no model folder resolved in this environment");
        return None;
    };
    let started = Instant::now();
    let engine = OnnxParakeetEngine::load(model_id, folder).expect("engine load");
    println!("OnnxParakeetEngine::load: {:?}", started.elapsed());
    Some((engine, load_wav_16k_mono_f32(&path).expect("parse jfk.wav")))
}

#[test]
fn jfk_transcribes_to_the_quote() {
    let Some((engine, jfk)) = load_engine_and_jfk() else {
        return;
    };

    let started = Instant::now();
    let result = engine.transcribe(&jfk, Some("en")).expect("transcribe");
    println!("jfk.wav ({} samples): {:?}", jfk.len(), started.elapsed());

    assert!(
        result.full_text.to_lowercase().contains(FIRST_HALF),
        "got: {}",
        result.full_text
    );
}

#[test]
fn a_full_window_transcribes_within_one_call() {
    let Some((engine, jfk)) = load_engine_and_jfk() else {
        return;
    };
    let window_samples = vuho_stt_engine::bench_support::WINDOW_SAMPLES;
    let window: Vec<f32> = jfk.iter().copied().cycle().take(window_samples).collect();

    let started = Instant::now();
    let result = engine.transcribe(&window, Some("en")).expect("transcribe");
    println!("one full window: {:?}", started.elapsed());

    assert!(
        result.full_text.to_lowercase().contains(FIRST_HALF),
        "got: {}",
        result.full_text
    );
}

#[test]
fn jfk_repeated_three_times_has_no_seam_duplication() {
    let Some((engine, jfk)) = load_engine_and_jfk() else {
        return;
    };
    let samples = jfk.repeat(3);

    let started = Instant::now();
    let result = engine.transcribe(&samples, Some("en")).expect("transcribe");
    println!(
        "jfk.wav x3 ({} samples): {:?}",
        samples.len(),
        started.elapsed()
    );
    println!("{}", result.full_text);

    let lower = result.full_text.to_lowercase();
    assert_eq!(
        lower.matches(FIRST_HALF).count(),
        3,
        "first half of the quote: {}",
        result.full_text
    );
    assert_eq!(
        lower.matches(SECOND_HALF).count(),
        3,
        "second half of the quote: {}",
        result.full_text
    );
}
