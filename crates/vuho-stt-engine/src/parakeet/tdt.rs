//! TDT greedy decoder for Parakeet-TDT.
//!
//! Implements the normative algorithm from the plan verbatim:
//! outer-active / inner-blank loop with zero-duration guards,
//! per-frame emission cap, and token count cap.
//!
//! Pure module: no inference-runtime dependency — the [`TdtStep`] trait is
//! the seam for injecting a real backend or a test fake.

use crate::token::{TokenAt, MAX_EMISSIONS_PER_POSITION};
use crate::EngineError;

/// Blank token id (also used as SOS).
pub(crate) const BLANK: u32 = 8192;
/// Token logits per frame: 8192 vocabulary pieces plus blank.
pub(crate) const TOKEN_LOGITS: usize = 8193;
/// Logits per frame: [`TOKEN_LOGITS`] token logits, then 5 duration bins.
pub(crate) const LOGITS_LEN: usize = 8198;
/// Maximum tokens emitted per 15s window (degenerate-chunk guard).
const MAX_TOKENS_PER_WINDOW: usize = 150;
/// Encoder feature dimension.
const ENCODER_DIM: usize = 1024;

/// One TDT model, as the greedy loop drives it: score an encoder frame,
/// and learn which token (if any) was emitted.
///
/// The split between `score` and `accept` is what lets a backend whose
/// decoder and joint are separate graphs (`CoreML` Parakeet) and one whose
/// decoder and joint are fused into a single graph (the ONNX export) sit
/// behind the same loop:
///
/// - `score` never changes what the next `score` sees. A fused backend
///   computes its next recurrent state while scoring; it must park that
///   candidate inside `State` rather than commit it.
/// - `accept` is called only for a non-blank token, and is the only thing
///   that advances the prediction network: after it, the next `score` is
///   conditioned on that token.
/// - On a blank frame nothing but `score` is called, so the state does not
///   advance. Stepping the prediction network on blank locks the decoder
///   into blank (ADR-015).
///
/// `pub` (not `pub(crate)`): re-exported by `bench_support` for
/// `benches/hot_paths.rs`.
pub trait TdtStep {
    /// The prediction-network state the loop threads through `score` and
    /// `accept`.
    type State;

    /// A fresh state, conditioned on the start-of-sequence blank.
    ///
    /// # Errors
    ///
    /// Returns the backend's error if priming the prediction network fails.
    fn start(&self) -> Result<Self::State, EngineError>;

    /// Score one encoder frame against `state`, writing into a
    /// caller-owned `logits` buffer instead of returning a fresh `Vec`
    /// (WP9: hot-path allocation discipline — this fires once per encoder
    /// frame visited, the hottest allocation in the decode loop). `logits`
    /// is cleared and repopulated with 8198 values: `[0..8193)` are token
    /// logits (vocab + blank), `[8193..8198)` are duration logits (5 bins).
    ///
    /// # Errors
    ///
    /// Returns the backend's error if inference fails.
    fn score(
        &self,
        enc_frame: &[f32],
        state: &mut Self::State,
        logits: &mut Vec<f32>,
    ) -> Result<(), EngineError>;

    /// Commit `token` (never blank) to `state`: the next `score` is
    /// conditioned on it.
    ///
    /// # Errors
    ///
    /// Returns the backend's error if inference fails.
    fn accept(&self, token: u32, state: &mut Self::State) -> Result<(), EngineError>;
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
/// `state` is owned by the caller — create it with [`TdtStep::start`] — and
/// may thread across calls over the same audio; this function never primes
/// or resets it.
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
/// Returns the error of the first `step` call that fails.
pub fn tdt_greedy<S: TdtStep>(
    enc: &[f32],
    enc_len: usize,
    initial_t: usize,
    global_frame_offset: usize,
    state: &mut S::State,
    step: &S,
) -> Result<(Vec<TokenAt>, usize), EngineError> {
    let mut t = initial_t;
    let mut emitted_at_t: usize = 0;
    let mut window_tokens: usize = 0;
    let mut emitted: Vec<TokenAt> = Vec::new();
    // Reused across every frame in this call instead of a fresh
    // ~8198-element `Vec` per frame (WP9) — `TdtStep::score` clears and
    // repopulates it each call, so its capacity is allocated at most once
    // per `tdt_greedy` invocation and reused for every subsequent frame.
    let mut logits: Vec<f32> = Vec::new();

    while t < enc_len {
        let frame_start = t * ENCODER_DIM;
        let enc_frame = &enc[frame_start..frame_start + ENCODER_DIM];

        step.score(enc_frame, state, &mut logits)?;
        debug_assert_eq!(
            logits.len(),
            LOGITS_LEN,
            "a frame scores {LOGITS_LEN} logits"
        );

        let tok = argmax_f32(&logits[..TOKEN_LOGITS]);
        let dur = argmax_f32(&logits[TOKEN_LOGITS..LOGITS_LEN]); // dur ∈ 0..=4

        if tok != BLANK {
            emitted.push(TokenAt {
                id: tok,
                pos: t + global_frame_offset,
            });
            step.accept(tok, state)?;
            emitted_at_t += 1;
            window_tokens += 1;
        }

        if dur > 0 {
            t += dur as usize;
            emitted_at_t = 0;
        } else if tok == BLANK || emitted_at_t >= MAX_EMISSIONS_PER_POSITION {
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
///
/// Shared with the voz backend's decode loop, which scores the same token
/// and duration heads (CONSTITUTION rule 26).
pub(crate) fn argmax_f32(slice: &[f32]) -> u32 {
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::bench_support::{fixed_step_model, write_scripted_logits};

    fn run_fixed(
        token: u32,
        duration: u32,
        enc_len: usize,
        initial_t: usize,
        global_frame_offset: usize,
    ) -> (Vec<TokenAt>, usize) {
        tdt_greedy(
            &vec![0.0; enc_len * ENCODER_DIM],
            enc_len,
            initial_t,
            global_frame_offset,
            &mut (),
            &fixed_step_model(token, duration),
        )
        .unwrap()
    }

    /// Blank at every frame: should advance t by 1 each step until `enc_len`.
    #[test]
    fn blank_every_frame_advances_one_per_step() {
        let enc_len = 10;

        let (emitted, next_t) = run_fixed(BLANK, 0, enc_len, 0, 0);

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

        let (emitted, next_t) = run_fixed(1, 2, enc_len, 0, 0);

        // Token 1 emitted at frames 0, 2, 4, 6, 8 → 5 times.
        assert_eq!(emitted.len(), 5);
        assert_eq!(emitted[0].pos, 0);
        assert_eq!(emitted[1].pos, 2);
        assert_eq!(emitted[4].pos, 8);
        assert_eq!(next_t, enc_len);
    }

    /// Duration=0 with non-blank: should stay at the same frame (re-emit),
    /// but the per-frame cap (10) eventually forces t+1.
    #[test]
    fn dur_zero_non_blank_caps_at_emission_limit() {
        let enc_len = 10;

        let (emitted, next_t) = run_fixed(1, 0, enc_len, 0, 0);

        // Each frame: emit token 1 ten times (cap), then t+1 → 10 frames × 10 = 100.
        assert_eq!(emitted.len(), 100);
        for (i, tok) in emitted.iter().enumerate() {
            assert_eq!(tok.pos, i / 10);
        }
        assert_eq!(next_t, enc_len);
    }

    /// Global frame offset is added to all emitted frames.
    #[test]
    fn global_frame_offset_is_added() {
        // One frame in the window, and any positive duration ends the loop
        // after the single emission — isolates the offset-addition logic.
        let (emitted, _next_t) = run_fixed(1, 1, 1, 0, 42);

        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].pos, 42); // 0 + 42
    }

    /// Token cap 150 breaks the loop.
    #[test]
    fn token_cap_breaks_loop() {
        let enc_len = 1000; // large enough that the cap hits first

        let (emitted, _next_t) = run_fixed(1, 0, enc_len, 0, 0);

        // 10 emissions per frame; the 151st emission (start of the 16th
        // frame) trips `window_tokens > 150` and breaks.
        assert_eq!(emitted.len(), 151);
    }

    /// `next_t` is the raw loop-exit position, not clamped to `enc_len`:
    /// decoder overshoots the window when duration jumps land past the end.
    #[test]
    fn next_t_reflects_raw_overshoot_past_enc_len() {
        let enc_len = 10;

        let (_emitted, next_t) = run_fixed(1, 3, enc_len, 0, 0);

        // Frames: 0, 3, 6, 9, 12 → loop exits at t=12 (enc_len=10).
        assert_eq!(next_t, 12);
    }

    /// Initial `t` carries forward: starting past `enc_len` yields an empty
    /// window immediately, with `next_t` unchanged from `initial_t` (the
    /// loop body never runs).
    #[test]
    fn initial_t_past_enc_len_yields_empty_window() {
        let (emitted, next_t) = run_fixed(1, 1, 5, 7, 0);

        assert!(emitted.is_empty());
        assert_eq!(next_t, 7);
    }

    #[derive(Debug, PartialEq, Eq)]
    enum Call {
        Start,
        Score { frame: usize, context: Vec<u32> },
        Accept(u32),
    }

    /// Records every call the loop makes. The prediction-network state is
    /// the list of tokens accepted so far; `decide` picks the token and
    /// duration for a frame from that state, like a real model would.
    struct Recording<F: Fn(usize, &[u32]) -> (u32, u32)> {
        decide: F,
        calls: RefCell<Vec<Call>>,
    }

    impl<F: Fn(usize, &[u32]) -> (u32, u32)> Recording<F> {
        fn new(decide: F) -> Self {
            Self {
                decide,
                calls: RefCell::new(Vec::new()),
            }
        }

        fn calls(self) -> Vec<Call> {
            self.calls.into_inner()
        }
    }

    impl<F: Fn(usize, &[u32]) -> (u32, u32)> TdtStep for Recording<F> {
        type State = Vec<u32>;

        fn start(&self) -> Result<Self::State, EngineError> {
            self.calls.borrow_mut().push(Call::Start);
            Ok(Vec::new())
        }

        fn score(
            &self,
            enc_frame: &[f32],
            state: &mut Self::State,
            logits: &mut Vec<f32>,
        ) -> Result<(), EngineError> {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let frame = enc_frame[0] as usize;
            self.calls.borrow_mut().push(Call::Score {
                frame,
                context: state.clone(),
            });
            let (token, duration) = (self.decide)(frame, state);
            write_scripted_logits(token, duration, logits);
            Ok(())
        }

        fn accept(&self, token: u32, state: &mut Self::State) -> Result<(), EngineError> {
            self.calls.borrow_mut().push(Call::Accept(token));
            state.push(token);
            Ok(())
        }
    }

    /// An encoder output whose every feature of frame `t` is `t`, so a fake
    /// can tell which frame it was asked to score.
    #[allow(clippy::cast_precision_loss)]
    fn numbered_frames(enc_len: usize) -> Vec<f32> {
        (0..enc_len)
            .flat_map(|t| std::iter::repeat_n(t as f32, ENCODER_DIM))
            .collect()
    }

    fn score(frame: usize, context: &[u32]) -> Call {
        Call::Score {
            frame,
            context: context.to_vec(),
        }
    }

    #[test]
    fn a_blank_frame_asks_the_model_to_score_and_nothing_else() {
        let model = Recording::new(|_, _| (BLANK, 1));
        let mut state = model.start().unwrap();

        let (emitted, _) = tdt_greedy(&numbered_frames(3), 3, 0, 0, &mut state, &model).unwrap();

        assert!(emitted.is_empty());
        assert_eq!(
            model.calls(),
            [Call::Start, score(0, &[]), score(1, &[]), score(2, &[])]
        );
    }

    #[test]
    fn an_emitted_token_is_accepted_before_the_next_score_and_conditions_it() {
        let model = Recording::new(|frame, context| match (frame, context.len()) {
            (0, 0) => (5, 0),
            (0, 1) => (7, 1),
            _ => (BLANK, 1),
        });
        let mut state = model.start().unwrap();

        let (emitted, _) = tdt_greedy(&numbered_frames(2), 2, 0, 0, &mut state, &model).unwrap();

        assert_eq!(
            emitted,
            [TokenAt { id: 5, pos: 0 }, TokenAt { id: 7, pos: 0 }]
        );
        assert_eq!(
            model.calls(),
            [
                Call::Start,
                score(0, &[]),
                Call::Accept(5),
                score(0, &[5]),
                Call::Accept(7),
                score(1, &[5, 7]),
            ]
        );
    }

    #[test]
    fn blank_frames_between_emissions_do_not_advance_the_state() {
        let model = Recording::new(|frame, _| if frame == 0 { (5, 1) } else { (BLANK, 1) });
        let mut state = model.start().unwrap();

        tdt_greedy(&numbered_frames(4), 4, 0, 0, &mut state, &model).unwrap();

        assert_eq!(state, [5]);
        assert_eq!(
            model.calls(),
            [
                Call::Start,
                score(0, &[]),
                Call::Accept(5),
                score(1, &[5]),
                score(2, &[5]),
                score(3, &[5]),
            ]
        );
    }

    #[test]
    fn the_loop_never_starts_or_resets_the_callers_state() {
        let model = Recording::new(|frame, _| if frame == 0 { (5, 1) } else { (BLANK, 1) });
        let mut state = vec![9];

        tdt_greedy(&numbered_frames(2), 2, 0, 0, &mut state, &model).unwrap();
        let (_, next_t) = tdt_greedy(&numbered_frames(2), 2, 1, 0, &mut state, &model).unwrap();

        assert_eq!(next_t, 2);
        assert_eq!(state, [9, 5]);
        assert_eq!(
            model.calls(),
            [
                score(0, &[9]),
                Call::Accept(5),
                score(1, &[9, 5]),
                score(1, &[9, 5]),
            ]
        );
    }

    /// A backend whose decoder and joint are one graph: scoring a frame
    /// runs the recurrent step on the last token and yields the next
    /// recurrent state as a by-product. It parks that candidate and only
    /// commits it when a token is accepted.
    struct Fused<F: Fn(usize, u32) -> (u32, u32)> {
        decide: F,
    }

    struct FusedState {
        last_token: u32,
        steps_committed: usize,
        parked_steps: usize,
    }

    impl<F: Fn(usize, u32) -> (u32, u32)> TdtStep for Fused<F> {
        type State = FusedState;

        fn start(&self) -> Result<Self::State, EngineError> {
            Ok(FusedState {
                last_token: BLANK,
                steps_committed: 0,
                parked_steps: 0,
            })
        }

        fn score(
            &self,
            enc_frame: &[f32],
            state: &mut Self::State,
            logits: &mut Vec<f32>,
        ) -> Result<(), EngineError> {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let frame = enc_frame[0] as usize;
            let (token, duration) = (self.decide)(frame, state.last_token);
            state.parked_steps = state.steps_committed + 1;
            write_scripted_logits(token, duration, logits);
            Ok(())
        }

        fn accept(&self, token: u32, state: &mut Self::State) -> Result<(), EngineError> {
            state.last_token = token;
            state.steps_committed = state.parked_steps;
            Ok(())
        }
    }

    #[test]
    fn a_fused_backend_parking_its_candidate_in_score_decodes_like_a_split_one() {
        let script = |frame: usize, last_token: u32| match (frame, last_token) {
            (0, BLANK) => (5, 0),
            (0, 5) => (7, 2),
            (2, _) => (9, 1),
            _ => (BLANK, 1),
        };
        let fused = Fused { decide: script };
        let mut state = fused.start().unwrap();

        let (emitted, next_t) =
            tdt_greedy(&numbered_frames(5), 5, 0, 0, &mut state, &fused).unwrap();

        assert_eq!(
            emitted,
            [
                TokenAt { id: 5, pos: 0 },
                TokenAt { id: 7, pos: 0 },
                TokenAt { id: 9, pos: 2 },
            ]
        );
        assert_eq!(next_t, 5);
        assert_eq!(
            state.steps_committed, 3,
            "one recurrent step per accepted token"
        );
    }
}
