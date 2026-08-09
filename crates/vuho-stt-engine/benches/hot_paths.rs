//! Model-free, deterministic benches for the hottest pure-Rust paths in
//! `vuho-stt-engine`'s decode/merge/window pipeline (WP9). Every benched
//! function is reached through `vuho_stt_engine::bench_support` (see that
//! module's doc comment for why a shim layer is needed at all):
//! `tdt_greedy` with a `FixedStepModel` fixture (no real `CoreML` call),
//! `merge::merge` + `segment_words` (indirectly, via `merge`) on synthetic
//! overlapping token runs, and `windower::plan`.
//!
//! Run: `cargo bench -p vuho-stt-engine`.

// `criterion_group!`/`criterion_main!` generate an undocumented `main` and
// a benches-registration function — same rationale as vuho-ui's
// `gpui::actions!` allow (WP2): no way to attach doc comments through the
// macro, and the names are self-explanatory criterion boilerplate.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vuho_stt_engine::bench_support::{
    fixed_step_model, merge, plan, tdt_greedy, token_at, DecoderState, MergeBounds, TokenAt,
    OVERLAP_FRAMES,
};

/// Encoder feature dimension (matches `parakeet::models::ENCODER_DIM` —
/// duplicated here since that constant isn't part of the bench surface;
/// the bench only needs a same-shaped synthetic encoder output, not the
/// real value's provenance).
const ENCODER_DIM: usize = 1024;

/// Encoder frames in one full 15 s model window: `ceil(WINDOW_SAMPLES /
/// SAMPLES_PER_FRAME) = ceil(240_000 / 1280) = ceil(187.5) = 188` (see
/// `windower`'s own doc comment on the same rounding). `jfk.wav` (11 s,
/// under the 15 s window size) is transcribed as exactly one such
/// window in `test-stt-ffi`'s batch path, so this is the real encoder
/// frame count that path exercises — not `jfk.wav`'s raw
/// `duration_s * (16_000 / 1280) ≈ 137.5` frame count, which is smaller
/// because the model always processes a full padded window.
const JFK_AUDIO_FRAMES: usize = 188;

/// The merge bounds Parakeet supplies (`WindowInference::merge_bounds`),
/// restated here because that method needs a loaded model this bench
/// deliberately does not have — `OVERLAP_FRAMES` itself still comes from
/// the crate.
const PARAKEET_MERGE_BOUNDS: MergeBounds = MergeBounds {
    search: OVERLAP_FRAMES,
    tolerance: OVERLAP_FRAMES / 2,
};

/// Vocabulary size used to build a varied synthetic token stream for the
/// `merge` bench — arbitrary but fixed so the id-cycling in
/// `synthetic_tokens` is deterministic and reproducible across runs.
const SYNTHETIC_VOCAB_SIZE: usize = 37;

/// A synthetic encoder output: `frames` frames of `ENCODER_DIM` zeros —
/// `tdt_greedy` never reads its content when driven by `FixedStepModel`
/// (which returns a fixed token/duration regardless of the frame), so an
/// all-zero buffer of the right length exercises the loop/emission-cap
/// logic identically to a real encoder output would, for benching
/// purposes.
fn synthetic_encoder_output(frames: usize) -> Vec<f32> {
    vec![0.0; frames * ENCODER_DIM]
}

/// A synthetic token run: `count` tokens starting at `frame` and advancing
/// one frame per token, ids cycling through a small vocabulary — enough
/// variety for `merge`'s word-segmentation/matching logic to do real work.
fn synthetic_tokens(count: usize, start_frame: usize) -> Vec<TokenAt> {
    (0..count)
        .map(|i| {
            token_at(
                (i % SYNTHETIC_VOCAB_SIZE).try_into().unwrap_or(0),
                start_frame + i,
            )
        })
        .collect()
}

/// Precomputed piece text for every synthetic vocabulary id, built once
/// (not per `piece(id)` call) so `synthetic_piece` can borrow from it —
/// `merge::merge`'s real `piece` closure (`ParakeetModels::piece_info`)
/// borrows from `self`'s already-loaded vocabulary the same way (WP9: no
/// per-token `String` allocation), and a bench that instead allocated a
/// fresh `String` on every `b.iter()` call would measure allocator noise,
/// not the code under test.
static SYNTHETIC_PIECES: std::sync::LazyLock<Vec<String>> = std::sync::LazyLock::new(|| {
    (0..SYNTHETIC_VOCAB_SIZE)
        .map(|id| format!("piece{id}"))
        .collect()
});

/// The same `piece` closure shape `merge::merge`'s real caller
/// (`ParakeetModels::piece_info`) uses: alternate word-initial/continuation
/// tokens on a fixed cadence, so `segment_words` produces multiple
/// same-length words to search over — a closer stand-in for real vocabulary
/// behavior than "every token is its own word."
// `merge`'s real callers (`ParakeetModels::piece_info`) return `None` for
// ids with no vocabulary entry; every synthetic id here has one, so this
// closure always returns `Some` — kept as `Option` to match `merge`'s
// actual closure signature (`impl Fn(u32) -> Option<(bool, &str)>`), not
// because this bench needs the `None` case.
#[allow(clippy::unnecessary_wraps)]
fn synthetic_piece(id: u32) -> Option<(bool, &'static str)> {
    let is_word_initial = id.is_multiple_of(3);
    Some((is_word_initial, SYNTHETIC_PIECES[id as usize].as_str()))
}

fn bench_tdt_greedy(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdt_greedy");
    for enc_len in [10usize, 100, JFK_AUDIO_FRAMES] {
        group.bench_function(format!("enc_len_{enc_len}_blank"), |b| {
            let enc = synthetic_encoder_output(enc_len);
            // 8192 = BLANK (parakeet::tdt::BLANK) — every frame decodes to
            // blank, exercising the pure time-advance loop without any
            // token emission.
            let model = fixed_step_model(8192, 0);
            b.iter(|| {
                let mut state = DecoderState::new();
                let result = tdt_greedy(
                    black_box(&enc),
                    black_box(enc_len),
                    black_box(0),
                    black_box(0),
                    &mut state,
                    &model,
                );
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge");
    for token_count in [25usize, 100, 400] {
        group.bench_function(format!("tokens_{token_count}"), |b| {
            let committed = synthetic_tokens(token_count, 0);
            b.iter(|| {
                // fresh overlaps the tail of committed by `OVERLAP_FRAMES`,
                // then continues past it — the realistic streaming shape
                // merge() is designed for.
                let fresh_start = committed.len().saturating_sub(OVERLAP_FRAMES);
                let fresh = synthetic_tokens(token_count, fresh_start);
                let outcome = merge(
                    black_box(&committed),
                    black_box(fresh),
                    black_box(PARAKEET_MERGE_BOUNDS),
                    synthetic_piece,
                );
                black_box(outcome)
            });
        });
    }
    group.finish();
}

fn bench_windower_plan(c: &mut Criterion) {
    let mut group = c.benchmark_group("windower_plan");
    // 11s (single window, jfk.wav's real length), 60s (multi-window), 300s
    // (many windows) at 16kHz.
    for seconds in [11usize, 60, 300] {
        let total_len = seconds * 16_000;
        group.bench_function(format!("{seconds}s"), |b| {
            b.iter(|| black_box(plan(black_box(total_len))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tdt_greedy, bench_merge, bench_windower_plan);
criterion_main!(benches);
