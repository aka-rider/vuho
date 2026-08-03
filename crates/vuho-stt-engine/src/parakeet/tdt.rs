//! TDT greedy decoder for Parakeet-TDT.
//!
//! Implements the normative algorithm from the plan verbatim:
//! outer-active / inner-blank loop with zero-duration guards,
//! per-frame emission cap, and token count cap.
//!
//! Pure module: no `CoreML` dependency — the `StepModel` trait is the
//! seam for injecting a real `CoreML` impl or a test fake.

use crate::EngineError;

use super::decoder_state::DecoderState;

/// Blank token id (also used as SOS).
pub(crate) const BLANK: u32 = 8192;
/// Maximum symbols emitted per frame before forcing t+1.
const MAX_SYMBOLS_PER_FRAME: usize = 10;
/// Maximum tokens emitted per 15s window (degenerate-chunk guard).
const MAX_TOKENS_PER_WINDOW: usize = 150;
/// Duration of one encoder frame in milliseconds (1280 samples @ 16kHz = 80ms).
pub(crate) const FRAME_MS: usize = 80;
/// Encoder feature dimension.
const ENCODER_DIM: usize = 1024;

/// Convert a global encoder frame index to milliseconds.
///
/// The one place this conversion happens (CONSTITUTION rule 26) — both the
/// batch window loop (`engine.rs`) and the streaming accumulator
/// (`stream::accumulator`) call this rather than keeping their own copies.
pub(crate) fn frame_ms(frame: usize) -> u64 {
    (frame * FRAME_MS) as u64
}

/// A token emitted during decoding, with its global frame index.
///
/// `pub` (not `pub(crate)`): re-exported by `bench_support` for
/// `benches/hot_paths.rs` — fields stay `pub(crate)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenAt {
    /// Token id in the vocabulary.
    pub(crate) id: u32,
    /// Global encoder frame index (includes window offset).
    pub(crate) frame: usize,
}

/// Trait for the decoder + joint models used by the TDT greedy loop.
///
/// Implementations call `CoreML` models (production) or return fixed
/// tables (tests).
///
/// `pub` (not `pub(crate)`): re-exported by `bench_support` for
/// `benches/hot_paths.rs`.
pub trait StepModel {
    /// Run one decoder step: given `token`, update `state` in place
    /// (h, c, `dec_out`).
    ///
    /// Called only for non-blank tokens — the loop never steps the
    /// LSTM on blank.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::CoreMl` if the `CoreML` call fails.
    fn decode(&self, token: i32, state: &mut DecoderState) -> Result<(), EngineError>;

    /// Compute joint logits for one encoder frame, writing into a
    /// caller-owned `out` buffer instead of returning a fresh `Vec` (WP9:
    /// hot-path allocation discipline — this fires once per encoder frame
    /// visited, the hottest allocation in the decode loop). `out` is
    /// cleared and repopulated with 8198 logits: `[0..8193)` are token
    /// logits (vocab + blank), `[8193..8198)` are duration logits (5 bins).
    ///
    /// # Errors
    ///
    /// Returns `EngineError::CoreMl` if the `CoreML` call fails.
    fn joint(
        &self,
        enc_frame: &[f32],
        dec_out: &[f32],
        out: &mut Vec<f32>,
    ) -> Result<(), EngineError>;
}

/// Run the TDT greedy decode loop on an encoder output tensor.
///
/// `pub` (not `pub(crate)`): re-exported by `bench_support` for
/// `benches/hot_paths.rs`.
///
/// `enc` is the flat encoder output: `enc_len` frames of `ENCODER_DIM`
/// each, row-major (`enc[t * ENCODER_DIM .. (t+1) * ENCODER_DIM]` is frame
/// `t`). `enc_len` is the actual (non-padded) length (≤ 188 for the 15s
/// model). `initial_t` is the starting frame index. `global_frame_offset`
/// adds to all emitted frame indices.
///
/// `state` threads across calls within a session: this function primes
/// the decoder (one `decode(BLANK, state)` call) only the first time it is
/// invoked with an unprimed state (`state.dec_out.is_none()`), matching the
/// "once per session, not per window" contract in the plan.
///
/// Returns `(emitted_tokens, next_t)` where `next_t` is the **raw**
/// loop-exit frame index (`t` — not `t - enc_len`). Callers doing a
/// same-window re-inference (a later call over a *longer* prefix of the
/// same audio, e.g. after a streaming VAD-endpoint promotion) pass
/// `next_t` straight back in as the next call's `initial_t`, so the
/// decoder never re-walks frames a state update already accounts for.
/// Callers advancing to a genuinely new window (a disjoint or
/// overlap-shifted encoder output) instead derive `time_jump =
/// next_t.saturating_sub(enc_len)` themselves before combining it with
/// that window's own boundary logic.
///
/// # Errors
///
/// Returns `EngineError::CoreMl` if any model call fails.
///
/// # Panics
///
/// Panics only if `state.dec_out` is still `None` immediately after the
/// priming `decode` call above — i.e. only on a `StepModel::decode`
/// implementation bug (a real implementation always sets `dec_out`); never
/// under normal operation.
pub fn tdt_greedy(
    enc: &[f32],
    enc_len: usize,
    initial_t: usize,
    global_frame_offset: usize,
    state: &mut DecoderState,
    m: &impl StepModel,
) -> Result<(Vec<TokenAt>, usize), EngineError> {
    if state.dec_out.is_none() {
        // Prime: one decode step with the current last_token (BLANK/SOS on
        // a fresh session) populates h/c/dec_out for the loop below.
        m.decode(state.last_token, state)?;
    }

    let mut t = initial_t;
    let mut emitted_at_t: usize = 0;
    let mut window_tokens: usize = 0;
    let mut emitted: Vec<TokenAt> = Vec::new();
    // Reused across every frame in this call instead of a fresh
    // ~8198-element `Vec` per frame (WP9) — `StepModel::joint` clears and
    // repopulates it each call, so its capacity is allocated at most once
    // per `tdt_greedy` invocation and reused for every subsequent frame.
    let mut logits: Vec<f32> = Vec::new();

    while t < enc_len {
        let frame_start = t * ENCODER_DIM;
        let enc_frame = &enc[frame_start..frame_start + ENCODER_DIM];
        let dec_out = state.dec_out.as_ref().expect("decoder primed above");

        m.joint(enc_frame, dec_out, &mut logits)?;
        debug_assert_eq!(logits.len(), 8198, "RNNTJoint logits must be 8198");

        let tok = argmax_f32(&logits[..8193]);
        let dur = argmax_f32(&logits[8193..8198]); // dur ∈ 0..=4

        if tok == BLANK {
            // Blank never steps the LSTM — dec_out/h/c stay as they were.
        } else {
            emitted.push(TokenAt {
                id: tok,
                frame: t + global_frame_offset,
            });
            #[allow(clippy::cast_possible_wrap)]
            let tok_i32 = tok as i32;
            state.last_token = tok_i32;
            m.decode(tok_i32, state)?;
            emitted_at_t += 1;
            window_tokens += 1;
        }

        if dur > 0 {
            t += dur as usize;
            emitted_at_t = 0;
        } else if tok == BLANK || emitted_at_t >= MAX_SYMBOLS_PER_FRAME {
            // Zero-duration blank, or a non-blank run that hit the
            // per-frame emission cap: force the frame to advance so a
            // degenerate joint output can never loop forever.
            t += 1;
            emitted_at_t = 0;
        }
        // else: non-blank with dur 0 under the cap → stay at t, re-emit.

        if window_tokens > MAX_TOKENS_PER_WINDOW {
            break;
        }
    }

    Ok((emitted, t))
}

/// Argmax over a slice, returning the index of the maximum value.
fn argmax_f32(slice: &[f32]) -> u32 {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| {
            // Slices are always the small, fixed vocab/duration ranges
            // (≤ 8198 entries), well within u32 range.
            #[allow(clippy::cast_possible_truncation)]
            let idx = i as u32;
            idx
        })
}

// ── Test helpers ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake `StepModel` that always emits a fixed `token` at a fixed
    /// `duration`, regardless of frame or decoder state — enough to drive
    /// the time-advance/emission-cap logic under test without a real model.
    struct FixedStepModel {
        token: u32,
        duration: u32,
    }

    impl StepModel for FixedStepModel {
        fn decode(&self, _token: i32, state: &mut DecoderState) -> Result<(), EngineError> {
            state.dec_out = Some(vec![0.0; 640]);
            Ok(())
        }

        fn joint(
            &self,
            _enc_frame: &[f32],
            _dec_out: &[f32],
            out: &mut Vec<f32>,
        ) -> Result<(), EngineError> {
            out.clear();
            out.resize(8198, f32::NEG_INFINITY);
            out[self.token as usize] = 0.0;
            for (i, bin) in out[8193..8198].iter_mut().enumerate() {
                #[allow(clippy::cast_possible_truncation)] // i is always 0..5
                let i = i as u32;
                *bin = if i == self.duration {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
            }
            Ok(())
        }
    }

    /// Blank at every frame: should advance t by 1 each step until `enc_len`.
    #[test]
    fn blank_every_frame_advances_one_per_step() {
        let enc_len = 10;
        let mut state = DecoderState::new();
        let model = FixedStepModel {
            token: BLANK,
            duration: 0,
        };

        let (emitted, next_t) = tdt_greedy(
            &vec![0.0; enc_len * ENCODER_DIM],
            enc_len,
            0,
            0,
            &mut state,
            &model,
        )
        .unwrap();

        assert!(emitted.is_empty(), "blank is never emitted as a token");
        assert_eq!(
            next_t, enc_len,
            "t advanced exactly to enc_len, no overshoot"
        );
    }

    /// Constant duration=2: should skip frames by 2.
    #[test]
    fn duration_two_skips_frames() {
        let enc_len = 10;
        let mut state = DecoderState::new();
        let model = FixedStepModel {
            token: 1,
            duration: 2,
        };

        let (emitted, next_t) = tdt_greedy(
            &vec![0.0; enc_len * ENCODER_DIM],
            enc_len,
            0,
            0,
            &mut state,
            &model,
        )
        .unwrap();

        // Token 1 emitted at frames 0, 2, 4, 6, 8 → 5 times.
        assert_eq!(emitted.len(), 5);
        assert_eq!(emitted[0].frame, 0);
        assert_eq!(emitted[1].frame, 2);
        assert_eq!(emitted[4].frame, 8);
        assert_eq!(next_t, enc_len);
    }

    /// Duration=0 with non-blank: should stay at the same frame (re-emit),
    /// but the per-frame cap (10) eventually forces t+1.
    #[test]
    fn dur_zero_non_blank_caps_at_emission_limit() {
        let enc_len = 10;
        let mut state = DecoderState::new();
        let model = FixedStepModel {
            token: 1,
            duration: 0,
        };

        let (emitted, next_t) = tdt_greedy(
            &vec![0.0; enc_len * ENCODER_DIM],
            enc_len,
            0,
            0,
            &mut state,
            &model,
        )
        .unwrap();

        // Each frame: emit token 1 ten times (cap), then t+1 → 10 frames × 10 = 100.
        assert_eq!(emitted.len(), 100);
        for (i, tok) in emitted.iter().enumerate() {
            assert_eq!(tok.frame, i / 10);
        }
        assert_eq!(next_t, enc_len);
    }

    /// Global frame offset is added to all emitted frames.
    #[test]
    fn global_frame_offset_is_added() {
        // One frame in the window, and any positive duration ends the loop
        // after the single emission — isolates the offset-addition logic.
        let enc_len = 1;
        let mut state = DecoderState::new();
        let model = FixedStepModel {
            token: 1,
            duration: 1,
        };

        let (emitted, _next_t) = tdt_greedy(
            &vec![0.0; enc_len * ENCODER_DIM],
            enc_len,
            0,
            42,
            &mut state,
            &model,
        )
        .unwrap();

        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].frame, 42); // 0 + 42
    }

    /// Token cap 150 breaks the loop.
    #[test]
    fn token_cap_breaks_loop() {
        let enc_len = 1000; // large enough that the cap hits first
        let mut state = DecoderState::new();
        let model = FixedStepModel {
            token: 1,
            duration: 0,
        };

        let (emitted, _next_t) = tdt_greedy(
            &vec![0.0; enc_len * ENCODER_DIM],
            enc_len,
            0,
            0,
            &mut state,
            &model,
        )
        .unwrap();

        // 10 emissions per frame; the 151st emission (start of the 16th
        // frame) trips `window_tokens > 150` and breaks.
        assert_eq!(emitted.len(), 151);
    }

    /// `next_t` is the raw loop-exit position, not clamped to `enc_len`:
    /// decoder overshoots the window when duration jumps land past the end.
    #[test]
    fn next_t_reflects_raw_overshoot_past_enc_len() {
        let enc_len = 10;
        let mut state = DecoderState::new();
        let model = FixedStepModel {
            token: 1,
            duration: 3,
        };

        let (_emitted, next_t) = tdt_greedy(
            &vec![0.0; enc_len * ENCODER_DIM],
            enc_len,
            0,
            0,
            &mut state,
            &model,
        )
        .unwrap();

        // Frames: 0, 3, 6, 9, 12 → loop exits at t=12 (enc_len=10).
        assert_eq!(next_t, 12);
    }

    /// State threads across two `tdt_greedy` calls: the decoder is primed
    /// only once (the second call reuses the already-primed `dec_out`).
    #[test]
    fn state_persists_across_calls_without_reprime() {
        let enc_len = 3;
        let mut state = DecoderState::new();
        let model = FixedStepModel {
            token: BLANK,
            duration: 1,
        };

        assert!(state.dec_out.is_none());
        let _ = tdt_greedy(
            &vec![0.0; enc_len * ENCODER_DIM],
            enc_len,
            0,
            0,
            &mut state,
            &model,
        )
        .unwrap();
        assert!(state.dec_out.is_some(), "priming sets dec_out");

        // A second call with the same (already-primed) state must not reset
        // dec_out to None or otherwise lose continuity.
        let dec_out_before = state.dec_out.clone();
        let _ = tdt_greedy(
            &vec![0.0; enc_len * ENCODER_DIM],
            enc_len,
            0,
            0,
            &mut state,
            &model,
        )
        .unwrap();
        assert_eq!(
            state.dec_out, dec_out_before,
            "blank-only window must not perturb dec_out"
        );
    }

    /// Initial `t` carries forward: starting past `enc_len` yields an empty
    /// window immediately, with `next_t` unchanged from `initial_t` (the
    /// loop body never runs).
    #[test]
    fn initial_t_past_enc_len_yields_empty_window() {
        let enc_len = 5;
        let mut state = DecoderState::new();
        let model = FixedStepModel {
            token: 1,
            duration: 1,
        };

        let (emitted, next_t) = tdt_greedy(
            &vec![0.0; enc_len * ENCODER_DIM],
            enc_len,
            7,
            0,
            &mut state,
            &model,
        )
        .unwrap();

        assert!(emitted.is_empty());
        assert_eq!(next_t, 7);
    }
}
