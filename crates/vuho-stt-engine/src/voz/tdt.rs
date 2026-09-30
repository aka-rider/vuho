//! The greedy TDT decode over voz's fused decoder.
//!
//! Pure: the decoder is a [`DecoderStep`], so the walk over the encoder
//! frames is tested without `CoreML`.
//!
//! One decoder call scores up to `decode_width` consecutive encoder frames
//! against a single prediction-network state. The walk reads those frames
//! left to right and stops at the first token; every frame it read before
//! that was blank, and so left the state as it was. A token starts a new
//! call — with the token as the new label and the state the call returned —
//! at the position the token's predicted duration reaches.

use crate::parakeet::tdt::argmax_f32;
use crate::token::{TokenAt, MAX_EMISSIONS_PER_POSITION};
use crate::voz::meta::Meta;
use crate::EngineError;

/// The prediction network's LSTM state, every layer stacked.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LstmState {
    pub(crate) h: Vec<f32>,
    pub(crate) c: Vec<f32>,
}

impl LstmState {
    fn zeros(meta: &Meta) -> Self {
        Self {
            h: vec![0.0; meta.state_len()],
            c: vec![0.0; meta.state_len()],
        }
    }
}

/// What one decoder call returns.
pub(crate) struct StepOutput {
    /// `span * n_logits` scores, frame after frame: frame `j`'s scores are
    /// `logits[j * n_logits..][..n_logits]` — the token ids, then the
    /// blank, then the duration bins.
    pub(crate) logits: Vec<f32>,
    /// The state after consuming the call's label.
    pub(crate) state: LstmState,
}

/// One fused prediction-network + joint call.
pub(crate) trait DecoderStep {
    /// Consume `label` from `state` and score the `span` encoder frames
    /// that start at `position`.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::CoreMl` if the underlying call fails.
    fn step(
        &self,
        label: u32,
        state: &LstmState,
        position: usize,
        span: usize,
    ) -> Result<StepOutput, EngineError>;
}

/// What the walk found in one call's frames.
enum Scan {
    /// A token at `offset` frames into the call, with its predicted
    /// duration in encoder frames.
    Token {
        id: u32,
        offset: usize,
        duration: usize,
    },
    /// Only blanks; the walk ran `offset` frames, which is at least the
    /// span.
    Blank { offset: usize },
}

/// Read one call's frames left to right until the first token, hopping
/// over each blank by its predicted duration (at least one frame).
fn scan(output: &StepOutput, span: usize, meta: &Meta) -> Scan {
    let blank = meta.blank_idx;
    let mut offset = 0;
    while offset < span {
        let frame = &output.logits[offset * meta.n_logits..][..meta.n_logits];
        let id = argmax_f32(&frame[..=blank as usize]);
        let duration = meta.durations[argmax_f32(&frame[blank as usize + 1..]) as usize];
        if id != blank {
            return Scan::Token {
                id,
                offset,
                duration,
            };
        }
        offset += duration.max(1);
    }
    Scan::Blank { offset }
}

/// Greedy decode of the first `limit` encoder frames of one window.
///
/// The prediction network starts from zeroed state with the blank as its
/// label. Each token is stamped `global_frame_offset` plus the encoder
/// frame it was read at.
///
/// # Errors
///
/// Propagates whatever [`DecoderStep::step`] returns.
pub(crate) fn greedy_decode(
    decoder: &impl DecoderStep,
    meta: &Meta,
    limit: usize,
    global_frame_offset: usize,
) -> Result<Vec<TokenAt>, EngineError> {
    let mut emitted = Vec::new();
    let mut state = LstmState::zeros(meta);
    let mut label = meta.blank_idx;
    let mut position = 0;
    let mut zero_run = 0;

    while position < limit {
        let span = meta.decode_width.min(limit - position);
        let output = decoder.step(label, &state, position, span)?;
        match scan(&output, span, meta) {
            Scan::Blank { offset } => {
                zero_run = 0;
                position += offset.max(1);
            }
            Scan::Token {
                id,
                offset,
                duration,
            } => {
                emitted.push(TokenAt {
                    id,
                    pos: global_frame_offset + position + offset,
                });
                state = output.state;
                label = id;
                position += token_advance(offset, duration, &mut zero_run);
            }
        }
    }
    Ok(emitted)
}

/// How far the walk moves after a token read `offset` frames into a call.
///
/// `zero_run` counts the zero-duration tokens since the position last
/// advanced; reaching [`MAX_EMISSIONS_PER_POSITION`] forces a step of one.
fn token_advance(offset: usize, duration: usize, zero_run: &mut usize) -> usize {
    if offset > 0 || duration > 0 {
        *zero_run = 0;
    }
    if duration > 0 {
        return offset + duration;
    }
    *zero_run += 1;
    if *zero_run >= MAX_EMISSIONS_PER_POSITION {
        *zero_run = 0;
        offset + 1
    } else {
        offset
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    const BLANK: u32 = 3;

    /// `(label, position, span, h[0])` of one decoder call.
    type Call = (u32, usize, usize, f32);

    /// A decoder driven by `script(label, frame) -> (token id, duration
    /// bin)`; its state counts the calls that consumed a label.
    struct Scripted<F: Fn(u32, usize) -> (u32, usize)> {
        script: F,
        calls: RefCell<Vec<Call>>,
    }

    impl<F: Fn(u32, usize) -> (u32, usize)> Scripted<F> {
        fn new(script: F) -> Self {
            Self {
                script,
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl<F: Fn(u32, usize) -> (u32, usize)> DecoderStep for Scripted<F> {
        fn step(
            &self,
            label: u32,
            state: &LstmState,
            position: usize,
            span: usize,
        ) -> Result<StepOutput, EngineError> {
            self.calls
                .borrow_mut()
                .push((label, position, span, state.h[0]));
            let meta = Meta::tiny();
            let mut logits = Vec::new();
            for lane in 0..span {
                let (id, duration_bin) = (self.script)(label, position + lane);
                let mut frame = vec![0.0; meta.n_logits];
                frame[id as usize] = 1.0;
                frame[meta.blank_idx as usize + 1 + duration_bin] = 1.0;
                logits.extend(frame);
            }
            let mut next = state.clone();
            next.h[0] += 1.0;
            Ok(StepOutput {
                logits,
                state: next,
            })
        }
    }

    fn spans(decoder: &Scripted<impl Fn(u32, usize) -> (u32, usize)>) -> Vec<(usize, usize)> {
        decoder
            .calls
            .borrow()
            .iter()
            .map(|&(_, position, span, _)| (position, span))
            .collect()
    }

    #[test]
    fn a_window_of_blanks_emits_nothing_and_walks_every_frame() {
        let decoder = Scripted::new(|_, _| (BLANK, 1));
        let tokens = greedy_decode(&decoder, &Meta::tiny(), 20, 0).expect("decode");
        assert!(tokens.is_empty());
        assert_eq!(spans(&decoder), [(0, 8), (8, 8), (16, 4)]);
    }

    /// The last call covers only what is left — the partial span of
    /// fewer than `decode_width` frames.
    #[test]
    fn the_last_call_spans_only_the_frames_that_remain() {
        let decoder = Scripted::new(|_, _| (BLANK, 1));
        greedy_decode(&decoder, &Meta::tiny(), 11, 0).expect("decode");
        assert_eq!(spans(&decoder), [(0, 8), (8, 3)]);
    }

    /// A blank whose duration runs past the call's frames moves the walk
    /// by the whole run, not by the span.
    #[test]
    fn a_blank_duration_hops_over_frames() {
        let decoder = Scripted::new(|_, _| (BLANK, 4));
        greedy_decode(&decoder, &Meta::tiny(), 20, 0).expect("decode");
        assert_eq!(spans(&decoder), [(0, 8), (8, 8), (16, 4)]);

        let short = Scripted::new(|_, _| (BLANK, 4));
        greedy_decode(&short, &Meta::tiny(), 5, 0).expect("decode");
        assert_eq!(spans(&short), [(0, 5)]);
    }

    /// A token restarts the walk at `frame + duration` with the token as
    /// the label and the state that call returned; frames read before it
    /// were blank and cost no state.
    #[test]
    fn a_token_reruns_the_decoder_from_the_new_position_with_updated_state() {
        let decoder = Scripted::new(|_, frame| if frame == 3 { (1, 2) } else { (BLANK, 1) });
        let tokens = greedy_decode(&decoder, &Meta::tiny(), 12, 100).expect("decode");

        assert_eq!(tokens, [TokenAt { id: 1, pos: 103 }]);
        assert_eq!(
            *decoder.calls.borrow(),
            [(BLANK, 0, 8, 0.0), (1, 5, 7, 1.0)],
            "the second call consumes the token and starts at 3 + 2"
        );
    }

    /// Ten zero-duration tokens in a row on one frame, then the walk is
    /// forced on by one — and can therefore not loop forever.
    #[test]
    fn ten_zero_duration_tokens_in_a_row_force_a_step_of_one() {
        let decoder = Scripted::new(|_, frame| if frame < 2 { (1, 0) } else { (BLANK, 1) });
        let tokens = greedy_decode(&decoder, &Meta::tiny(), 4, 0).expect("decode");

        let at = |frame| tokens.iter().filter(|t| t.pos == frame).count();
        assert_eq!((at(0), at(1), at(2)), (10, 10, 0));
    }

    /// Each zero-duration token here follows blank reads that moved the
    /// position, so the run never builds up and the walk is never forced
    /// past the token at frame 40.
    #[test]
    fn ten_zero_duration_tokens_separated_by_blanks_do_not_force_a_step() {
        let token_every_fourth_frame = |label: u32, frame: usize| {
            let token = if (frame / 4).is_multiple_of(2) { 1 } else { 2 };
            if frame.is_multiple_of(4) && label != token {
                (token, 0)
            } else {
                (BLANK, 4)
            }
        };
        let decoder = Scripted::new(token_every_fourth_frame);
        let tokens = greedy_decode(&decoder, &Meta::tiny(), 48, 0).expect("decode");

        let positions: Vec<usize> = tokens.iter().map(|t| t.pos).collect();
        assert_eq!(positions, (0..12).map(|n| n * 4).collect::<Vec<_>>());
    }
}
