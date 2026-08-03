//! Transcribe jfk.wav via the Parakeet-TDT STT engine.
//!
//! Reads the WAV file, initializes the engine, and verifies the transcription
//! contains the expected JFK quote.
//!
//! Gate: `cargo run -p test-stt-ffi` prints **PASS** when the transcription
//! contains the expected quote.
//!
//! WAV loading (path resolution + PCM decode) is NOT reimplemented here —
//! `vuho_stt_engine::test_support` (behind the `test-fixtures` feature, see
//! this crate's `Cargo.toml`) is the one place in the workspace that logic
//! lives (CONSTITUTION rule 26); this binary and
//! `vuho-stt-engine/tests/batch_multiwindow.rs` both call it.

use vuho_stt_engine::test_support::{jfk_wav_path, load_wav_16k_mono_f32};
use vuho_stt_engine::{ParakeetEngine, TranscriptionEngine};

/// Sample rate the Parakeet-TDT engine expects (16 kHz mono).
const STT_SAMPLE_RATE: u32 = 16_000;

fn main() {
    env_logger::init();

    let audio_path = jfk_wav_path().unwrap_or_else(|| {
        eprintln!(
            "ERROR: JFK_WAV/jfk.wav not found — set JFK_WAV or place jfk.wav at the workspace root"
        );
        std::process::exit(1);
    });

    // The engine's own chokepoint resolves the model (and honors VUHO_MODEL_FOLDER),
    // so this gate exercises exactly the path the app takes.
    let model_folder = vuho_stt_engine::resolve_model_folder().unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    println!("=== Vuho Parakeet-TDT STT Test ===");
    println!("Audio: {}", audio_path.display());
    println!("Model: {}", model_folder.display());
    println!();

    // ── Read WAV file ───────────────────────────────────────────────────
    let samples: Vec<f32> = load_wav_16k_mono_f32(&audio_path).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    #[allow(clippy::cast_precision_loss)]
    // display-only duration; sample counts never approach 2^52
    let duration_s = samples.len() as f64 / f64::from(STT_SAMPLE_RATE);
    println!("Loaded {} samples ({duration_s:.2}s)", samples.len());
    println!();

    // ── Initialize STT engine ───────────────────────────────────────────
    println!("Initializing engine and loading models...");
    let engine = ParakeetEngine::load(model_folder).expect("engine load");
    println!("Models loaded.");
    println!();

    // ── Transcribe ──────────────────────────────────────────────────────
    println!("Transcribing...");
    let result = engine.transcribe(&samples, Some("en")).expect("transcribe");

    println!("=== Transcription Result ===");
    println!("Language: {}", result.language);
    println!("Segments: {}", result.segments.len());
    for seg in &result.segments {
        #[allow(clippy::cast_precision_loss)]
        // display-only duration; ms timestamps never approach 2^52
        let (start_s, end_s) = (seg.start_ms as f64 / 1000.0, seg.end_ms as f64 / 1000.0);
        println!("  [{start_s:.2}s - {end_s:.2}s] {}", seg.text);
    }
    println!();
    println!("Full text: {}", result.full_text);
    println!();

    // ── Verify ──────────────────────────────────────────────────────────
    let lower = result.full_text.to_lowercase();
    let expected = "ask not what your country can do for you";

    if lower.contains(expected) {
        println!("PASS: Transcription contains expected quote!");
    } else {
        eprintln!("FAIL: Expected to find \"{expected}\" in transcription");
        eprintln!("Got: {}", result.full_text);
        std::process::exit(1);
    }

    // Clean up.
    engine.unload();
    println!("\n=== DONE ===");
}
