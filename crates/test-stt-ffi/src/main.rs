//! Transcribe jfk.wav via one of the workspace's STT backends.
//!
//! Reads the WAV file, loads the engine for the selected model, and verifies
//! the transcription contains the expected JFK quote.
//!
//! Gate: `cargo run -p test-stt-ffi` prints **PASS** for the manifest's
//! default model; `cargo run -p test-stt-ffi -- --model <id>` runs any other
//! model the manifest declares.
//!
//! WAV loading (path resolution + PCM decode) is NOT reimplemented here —
//! `vuho_stt_engine::test_support` (behind the `test-fixtures` feature, see
//! this crate's `Cargo.toml`) is the one place in the workspace that logic
//! lives (CONSTITUTION rule 26); this binary and
//! `vuho-stt-engine/tests/batch_multiwindow.rs` both call it.

use vuho_model_paths::Backend;
use vuho_stt_engine::test_support::{jfk_wav_path, load_wav_16k_mono_f32};
use vuho_stt_engine::{CanaryEngine, ParakeetEngine, TranscriptionEngine};

/// Sample rate every backend here expects (16 kHz mono).
const STT_SAMPLE_RATE: u32 = 16_000;

/// The quote jfk.wav contains — the gate's pass condition.
const EXPECTED_QUOTE: &str = "ask not what your country can do for you";

fn main() {
    env_logger::init();

    let model_id = selected_model_id();
    let model = vuho_model_paths::manifest()
        .stt
        .model(&model_id)
        .unwrap_or_else(|| {
            eprintln!("ERROR: unknown model id: {model_id}");
            std::process::exit(1);
        });

    let audio_path = jfk_wav_path().unwrap_or_else(|| {
        eprintln!(
            "ERROR: JFK_WAV/jfk.wav not found — set JFK_WAV or place jfk.wav at the workspace root"
        );
        std::process::exit(1);
    });

    // The engine's own chokepoint resolves the model (and honors VUHO_MODEL_FOLDER),
    // so this gate exercises exactly the path the app takes.
    let model_folder = vuho_stt_engine::resolve_model_folder(&model_id).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    println!("=== Vuho STT Test ===");
    println!("Model: {} ({model_id})", model.display_name);
    println!("Audio: {}", audio_path.display());
    println!("Folder: {}", model_folder.display());
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
    let engine: Box<dyn TranscriptionEngine> = match model.backend {
        Backend::ParakeetTdt => {
            Box::new(ParakeetEngine::load(&model_id, model_folder).expect("Parakeet engine load"))
        }
        Backend::CanaryAed => {
            Box::new(CanaryEngine::load(&model_id, model_folder).expect("Canary engine load"))
        }
    };
    println!("Models loaded.");
    println!();

    // ── Transcribe ──────────────────────────────────────────────────────
    println!("Transcribing...");
    let started = std::time::Instant::now();
    let result = engine.transcribe(&samples, Some("en")).expect("transcribe");
    let elapsed = started.elapsed();

    println!("=== Transcription Result ===");
    println!("Language: {}", result.language);
    println!("Transcribe wall clock: {elapsed:?}");
    println!("Segments: {}", result.segments.len());
    for (index, seg) in result.segments.iter().enumerate() {
        println!(
            "  {} {}",
            segment_label(model.backend, seg, index),
            seg.text
        );
    }
    println!();
    println!("Full text: {}", result.full_text);
    println!();

    // ── Verify ──────────────────────────────────────────────────────────
    let lower = result.full_text.to_lowercase();

    if lower.contains(EXPECTED_QUOTE) {
        println!("PASS: Transcription contains expected quote!");
    } else {
        eprintln!("FAIL: Expected to find \"{EXPECTED_QUOTE}\" in transcription");
        eprintln!("Got: {}", result.full_text);
        std::process::exit(1);
    }

    // Clean up.
    engine.unload();
    println!("\n=== DONE ===");
}

/// `--model <id>`, defaulting to the manifest's own default model.
fn selected_model_id() -> String {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => vuho_model_paths::manifest().stt.default_model.clone(),
        [flag, id] if flag == "--model" => id.clone(),
        _ => {
            eprintln!("ERROR: usage: test-stt-ffi [--model <model-id>]");
            std::process::exit(1);
        }
    }
}

/// A segment's diagnostic prefix.
///
/// A backend whose token positions are real encoder frames gets real
/// timestamps; one whose positions are a fixed synthetic stride gets its
/// segment index instead, so this printout never shows a plausible-looking
/// time that means nothing (CONSTITUTION rule 2).
fn segment_label(backend: Backend, seg: &vuho_domain::TranscriptSegment, index: usize) -> String {
    match backend {
        Backend::ParakeetTdt => {
            #[allow(clippy::cast_precision_loss)]
            // display-only duration; ms timestamps never approach 2^52
            let (start_s, end_s) = (seg.start_ms as f64 / 1000.0, seg.end_ms as f64 / 1000.0);
            format!("[{start_s:.2}s - {end_s:.2}s]")
        }
        Backend::CanaryAed => format!("[segment {index}]"),
    }
}
