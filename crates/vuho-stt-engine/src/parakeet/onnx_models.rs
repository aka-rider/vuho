//! Parakeet-TDT over ONNX Runtime: the preprocessor, encoder and fused
//! decoder+joint graphs of the ONNX export, and the [`TdtStep`] that drives
//! the fused graph.

use std::path::Path;
use std::time::Instant;

use crate::onnx::{Dim, ElementKind, ExpectedTensor, Input, OnnxSession};
use crate::stream::merge::MergeBounds;
use crate::stream::windower::WINDOW_SAMPLES;
use crate::token::TokenAt;
use crate::vocab::Vocab;
use crate::window_inference::WindowInference;
use crate::EngineError;

use super::tdt::{tdt_greedy, TdtStep, BLANK, ENCODER_DIM, LOGITS_LEN};

const MEL_BINS: usize = 128;
const PREDICTION_LAYERS: usize = 2;
const PREDICTION_HIDDEN: usize = 640;
const RECURRENT_LEN: usize = PREDICTION_LAYERS * PREDICTION_HIDDEN;

const PREPROCESSOR_INPUTS: &[ExpectedTensor] = &[
    ExpectedTensor {
        name: "waveforms",
        kind: ElementKind::F32,
        shape: &[Dim::Any, Dim::Any],
    },
    ExpectedTensor {
        name: "waveforms_lens",
        kind: ElementKind::I64,
        shape: &[Dim::Any],
    },
];
const PREPROCESSOR_OUTPUTS: &[ExpectedTensor] = &[
    ExpectedTensor {
        name: "features",
        kind: ElementKind::F32,
        shape: &[Dim::Any, Dim::Is(MEL_BINS), Dim::Any],
    },
    ExpectedTensor {
        name: "features_lens",
        kind: ElementKind::I64,
        shape: &[Dim::Any],
    },
];
const ENCODER_INPUTS: &[ExpectedTensor] = &[
    ExpectedTensor {
        name: "audio_signal",
        kind: ElementKind::F32,
        shape: &[Dim::Any, Dim::Is(MEL_BINS), Dim::Any],
    },
    ExpectedTensor {
        name: "length",
        kind: ElementKind::I64,
        shape: &[Dim::Any],
    },
];
const ENCODER_OUTPUTS: &[ExpectedTensor] = &[
    ExpectedTensor {
        name: "outputs",
        kind: ElementKind::F32,
        shape: &[Dim::Any, Dim::Is(ENCODER_DIM), Dim::Any],
    },
    ExpectedTensor {
        name: "encoded_lengths",
        kind: ElementKind::I64,
        shape: &[Dim::Any],
    },
];
const FUSED_INPUTS: &[ExpectedTensor] = &[
    ExpectedTensor {
        name: "encoder_outputs",
        kind: ElementKind::F32,
        shape: &[Dim::Any, Dim::Is(ENCODER_DIM), Dim::Any],
    },
    ExpectedTensor {
        name: "targets",
        kind: ElementKind::I32,
        shape: &[Dim::Any, Dim::Any],
    },
    ExpectedTensor {
        name: "target_length",
        kind: ElementKind::I32,
        shape: &[Dim::Any],
    },
    ExpectedTensor {
        name: "input_states_1",
        kind: ElementKind::F32,
        shape: &[
            Dim::Is(PREDICTION_LAYERS),
            Dim::Any,
            Dim::Is(PREDICTION_HIDDEN),
        ],
    },
    ExpectedTensor {
        name: "input_states_2",
        kind: ElementKind::F32,
        shape: &[
            Dim::Is(PREDICTION_LAYERS),
            Dim::Any,
            Dim::Is(PREDICTION_HIDDEN),
        ],
    },
];
const FUSED_OUTPUTS: &[ExpectedTensor] = &[
    ExpectedTensor {
        name: "outputs",
        kind: ElementKind::F32,
        shape: &[Dim::Any, Dim::Any, Dim::Any, Dim::Is(LOGITS_LEN)],
    },
    ExpectedTensor {
        name: "output_states_1",
        kind: ElementKind::F32,
        shape: &[
            Dim::Is(PREDICTION_LAYERS),
            Dim::Any,
            Dim::Is(PREDICTION_HIDDEN),
        ],
    },
    ExpectedTensor {
        name: "output_states_2",
        kind: ElementKind::F32,
        shape: &[
            Dim::Is(PREDICTION_LAYERS),
            Dim::Any,
            Dim::Is(PREDICTION_HIDDEN),
        ],
    },
];

/// One fused decoder+joint evaluation: the scores for an encoder frame, and
/// the recurrent state the prediction network reached by consuming the
/// input token.
pub(crate) struct JointStep {
    pub(crate) logits: Vec<f32>,
    pub(crate) h: Vec<f32>,
    pub(crate) c: Vec<f32>,
}

/// A graph that runs the prediction network and the joint in one call.
pub(crate) trait DecoderJoint {
    /// Consume `last_token` from recurrent state `(h, c)` and score
    /// `enc_frame` against the result.
    fn step(
        &self,
        enc_frame: &[f32],
        last_token: u32,
        h: &[f32],
        c: &[f32],
    ) -> Result<JointStep, EngineError>;
}

struct FusedGraph(OnnxSession);

impl DecoderJoint for FusedGraph {
    fn step(
        &self,
        enc_frame: &[f32],
        last_token: u32,
        h: &[f32],
        c: &[f32],
    ) -> Result<JointStep, EngineError> {
        let token = i32::try_from(last_token)
            .map_err(|_| EngineError::Transcribe(format!("token id {last_token} exceeds i32")))?;
        let recurrent_shape = vec![PREDICTION_LAYERS, 1, PREDICTION_HIDDEN];
        let mut outputs = self.0.run(
            vec![
                (
                    "encoder_outputs",
                    Input::F32(vec![1, ENCODER_DIM, 1], enc_frame.to_vec()),
                ),
                ("targets", Input::I32(vec![1, 1], vec![token])),
                ("target_length", Input::I32(vec![1], vec![1])),
                (
                    "input_states_1",
                    Input::F32(recurrent_shape.clone(), h.to_vec()),
                ),
                ("input_states_2", Input::F32(recurrent_shape, c.to_vec())),
            ],
            &["outputs", "output_states_1", "output_states_2"],
        )?;
        let (_, logits) = outputs.take_f32("outputs")?;
        let (_, h) = outputs.take_f32("output_states_1")?;
        let (_, c) = outputs.take_f32("output_states_2")?;
        if logits.len() != LOGITS_LEN || h.len() != RECURRENT_LEN || c.len() != RECURRENT_LEN {
            return Err(EngineError::Transcribe(format!(
                "fused decoder returned {} logits and {}/{} state values, expected {LOGITS_LEN} and {RECURRENT_LEN}",
                logits.len(),
                h.len(),
                c.len()
            )));
        }
        Ok(JointStep { logits, h, c })
    }
}

struct Recurrent {
    h: Vec<f32>,
    c: Vec<f32>,
}

impl Recurrent {
    fn zeroed() -> Self {
        Self {
            h: vec![0.0; RECURRENT_LEN],
            c: vec![0.0; RECURRENT_LEN],
        }
    }
}

/// What the fused step threads through a window: the committed prediction
/// state and the token it last consumed, plus the candidate state the most
/// recent `score` produced.
pub(crate) struct FusedState {
    committed: Recurrent,
    last_token: u32,
    parked: Option<Recurrent>,
}

/// [`TdtStep`] over a fused decoder+joint graph. Scoring a frame consumes
/// the committed last token, so the state it reaches is only a candidate:
/// `score` parks it and `accept` commits it.
pub(crate) struct FusedStep<'a, J> {
    joint: &'a J,
}

impl<J: DecoderJoint> TdtStep for FusedStep<'_, J> {
    type State = FusedState;

    fn start(&self) -> Result<FusedState, EngineError> {
        Ok(FusedState {
            committed: Recurrent::zeroed(),
            last_token: BLANK,
            parked: None,
        })
    }

    fn score(
        &self,
        enc_frame: &[f32],
        state: &mut FusedState,
        logits: &mut Vec<f32>,
    ) -> Result<(), EngineError> {
        let step = self.joint.step(
            enc_frame,
            state.last_token,
            &state.committed.h,
            &state.committed.c,
        )?;
        *logits = step.logits;
        state.parked = Some(Recurrent {
            h: step.h,
            c: step.c,
        });
        Ok(())
    }

    fn accept(&self, token: u32, state: &mut FusedState) -> Result<(), EngineError> {
        state.committed = state.parked.take().ok_or_else(|| {
            EngineError::Transcribe("a token was accepted without a preceding score".into())
        })?;
        state.last_token = token;
        Ok(())
    }
}

/// The loaded ONNX Parakeet-TDT graphs and vocabulary.
pub(crate) struct OnnxParakeetModels {
    preprocessor: OnnxSession,
    encoder: OnnxSession,
    decoder: FusedGraph,
    vocab: Vocab,
}

impl OnnxParakeetModels {
    /// Load every graph of `model_id` from `folder`, check each declares the
    /// signature this backend feeds it, and run one warm-up inference.
    ///
    /// # Errors
    ///
    /// `EngineError::LoadFailed` if the layout is incomplete, a graph's
    /// declared signature differs from what this backend expects, or the
    /// warm-up inference fails; `EngineError::Onnx` if ONNX Runtime rejects a
    /// file.
    pub(crate) fn load(model_id: &str, folder: &Path) -> Result<Self, EngineError> {
        crate::validate_model_layout(model_id, folder)?;
        let started = Instant::now();
        let threads = num_cpus::get_physical();
        let path = |role| crate::asset_path(model_id, role, folder);

        let preprocessor = OnnxSession::load(&path(crate::asset_role::PREPROCESSOR)?, threads)?;
        preprocessor.check_signature(PREPROCESSOR_INPUTS, PREPROCESSOR_OUTPUTS)?;
        let encoder = OnnxSession::load(&path(crate::asset_role::ENCODER)?, threads)?;
        encoder.check_signature(ENCODER_INPUTS, ENCODER_OUTPUTS)?;
        let decoder = OnnxSession::load(&path(crate::asset_role::DECODER)?, threads)?;
        decoder.check_signature(FUSED_INPUTS, FUSED_OUTPUTS)?;
        let vocab = Vocab::load(&path(crate::asset_role::VOCAB)?, Some(BLANK))?;

        let models = Self {
            preprocessor,
            encoder,
            decoder: FusedGraph(decoder),
            vocab,
        };
        models
            .infer_window(&vec![0.0; WINDOW_SAMPLES], 0)
            .map_err(|e| EngineError::LoadFailed(format!("warm-up inference failed: {e}")))?;
        log::info!(
            "parakeet-onnx: loaded and warmed up in {:?}",
            started.elapsed()
        );
        Ok(models)
    }

    fn infer_window(
        &self,
        samples: &[f32],
        global_frame_offset: usize,
    ) -> Result<Vec<TokenAt>, EngineError> {
        let (encoder_frames, frame_count) = self.encode(samples)?;
        let step = FusedStep {
            joint: &self.decoder,
        };
        let mut state = step.start()?;
        let (emitted, _) = tdt_greedy(
            &encoder_frames,
            frame_count,
            0,
            global_frame_offset,
            &mut state,
            &step,
        )?;
        Ok(emitted)
    }

    /// Preprocessor then encoder over the window at its real length;
    /// returns the valid frames as a row-major `frames × ENCODER_DIM`
    /// matrix, and the frame count.
    fn encode(&self, samples: &[f32]) -> Result<(Vec<f32>, usize), EngineError> {
        let length = i64::try_from(samples.len())
            .map_err(|_| EngineError::Transcribe("window length exceeds i64".into()))?;
        let mut mel = self.preprocessor.run(
            vec![
                (
                    "waveforms",
                    Input::F32(vec![1, samples.len()], samples.to_vec()),
                ),
                ("waveforms_lens", Input::I64(vec![1], vec![length])),
            ],
            &["features", "features_lens"],
        )?;
        let (mel_shape, features) = mel.take_f32("features")?;
        let mel_lengths = mel.take_i64("features_lens")?;

        let mut encoded = self.encoder.run(
            vec![
                ("audio_signal", Input::F32(mel_shape, features)),
                ("length", Input::I64(vec![1], mel_lengths)),
            ],
            &["outputs", "encoded_lengths"],
        )?;
        let (shape, channels_first) = encoded.take_f32("outputs")?;
        let lengths = encoded.take_i64("encoded_lengths")?;
        frames_from_channels_first(&shape, &channels_first, &lengths)
    }
}

/// Turn the encoder's `[1, ENCODER_DIM, T]` output into `frames × ENCODER_DIM`
/// row-major rows, keeping the `lengths[0]` valid frames.
fn frames_from_channels_first(
    shape: &[usize],
    channels_first: &[f32],
    lengths: &[i64],
) -> Result<(Vec<f32>, usize), EngineError> {
    let &[1, ENCODER_DIM, total] = shape else {
        return Err(EngineError::Transcribe(format!(
            "encoder output has shape {shape:?}, expected [1, {ENCODER_DIM}, T]"
        )));
    };
    let reported = lengths
        .first()
        .and_then(|len| usize::try_from(*len).ok())
        .ok_or_else(|| {
            EngineError::Transcribe(format!("encoder reported invalid lengths {lengths:?}"))
        })?;
    let frames = reported.min(total);
    let mut rows = Vec::with_capacity(frames * ENCODER_DIM);
    for frame in 0..frames {
        rows.extend((0..ENCODER_DIM).map(|channel| channels_first[channel * total + frame]));
    }
    Ok((rows, frames))
}

impl WindowInference for OnnxParakeetModels {
    fn infer_window(
        &self,
        samples: &[f32],
        global_frame_offset: usize,
        _language: &str,
    ) -> Result<Vec<TokenAt>, EngineError> {
        Self::infer_window(self, samples, global_frame_offset)
    }

    fn piece_info(&self, id: u32) -> Option<(bool, &str)> {
        self.vocab.piece_info(id)
    }

    fn detokenize(&self, tokens: &[TokenAt]) -> String {
        self.vocab.detokenize(tokens)
    }

    fn merge_bounds(&self) -> MergeBounds {
        MergeBounds::measured_positions()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::bench_support::write_scripted_logits;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < f32::EPSILON,
            "{actual} != {expected}"
        );
    }

    struct Call {
        last_token: u32,
        h_head: f32,
    }

    struct ScriptedJoint {
        token: u32,
        calls: RefCell<Vec<Call>>,
    }

    impl ScriptedJoint {
        fn new(token: u32) -> Self {
            Self {
                token,
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl DecoderJoint for ScriptedJoint {
        fn step(
            &self,
            _enc_frame: &[f32],
            last_token: u32,
            h: &[f32],
            c: &[f32],
        ) -> Result<JointStep, EngineError> {
            self.calls.borrow_mut().push(Call {
                last_token,
                h_head: h[0],
            });
            let mut logits = Vec::new();
            write_scripted_logits(self.token, 1, &mut logits);
            Ok(JointStep {
                logits,
                h: vec![h[0] + 1.0; RECURRENT_LEN],
                c: c.to_vec(),
            })
        }
    }

    fn score(step: &FusedStep<'_, ScriptedJoint>, state: &mut FusedState) {
        step.score(&[0.0; ENCODER_DIM], state, &mut Vec::new())
            .expect("scripted scoring succeeds");
    }

    #[test]
    fn a_fresh_state_is_conditioned_on_blank_with_zero_recurrence() {
        let joint = ScriptedJoint::new(BLANK);
        let step = FusedStep { joint: &joint };
        let mut state = step.start().expect("start needs no model call");

        score(&step, &mut state);

        let calls = joint.calls.borrow();
        assert_eq!(calls[0].last_token, BLANK);
        assert_close(calls[0].h_head, 0.0);
    }

    #[test]
    fn scoring_alone_never_advances_the_prediction_network() {
        let joint = ScriptedJoint::new(BLANK);
        let step = FusedStep { joint: &joint };
        let mut state = step.start().expect("start");

        score(&step, &mut state);
        score(&step, &mut state);

        let calls = joint.calls.borrow();
        assert_eq!(calls[1].last_token, BLANK);
        assert_close(calls[1].h_head, 0.0);
    }

    #[test]
    fn accepting_a_token_commits_the_state_its_own_frame_produced() {
        let joint = ScriptedJoint::new(7);
        let step = FusedStep { joint: &joint };
        let mut state = step.start().expect("start");

        score(&step, &mut state);
        step.accept(7, &mut state).expect("accept follows a score");
        score(&step, &mut state);

        let calls = joint.calls.borrow();
        assert_eq!(calls[1].last_token, 7);
        assert_close(calls[1].h_head, 1.0);
    }

    #[test]
    fn a_blank_after_an_accepted_token_keeps_the_committed_state() {
        let joint = ScriptedJoint::new(7);
        let step = FusedStep { joint: &joint };
        let mut state = step.start().expect("start");
        score(&step, &mut state);
        step.accept(7, &mut state).expect("accept");
        score(&step, &mut state);
        score(&step, &mut state);

        let calls = joint.calls.borrow();
        assert_eq!(calls[2].last_token, 7);
        assert_close(calls[2].h_head, 1.0);
    }

    #[test]
    fn accepting_without_a_preceding_score_is_an_error() {
        let joint = ScriptedJoint::new(7);
        let step = FusedStep { joint: &joint };
        let mut state = step.start().expect("start");

        let err = step.accept(7, &mut state).expect_err("nothing was scored");

        assert!(matches!(err, EngineError::Transcribe(_)));
    }

    #[test]
    fn the_greedy_loop_emits_through_the_fused_step() {
        let joint = ScriptedJoint::new(7);
        let step = FusedStep { joint: &joint };
        let mut state = step.start().expect("start");

        let (emitted, next_t) =
            tdt_greedy(&[0.0; 3 * ENCODER_DIM], 3, 0, 10, &mut state, &step).expect("decode");

        let positions: Vec<usize> = emitted.iter().map(|token| token.pos).collect();
        assert_eq!(positions, vec![10, 11, 12]);
        assert_eq!(next_t, 3);
    }

    #[test]
    fn the_encoder_output_is_transposed_into_frame_rows() {
        let total = 3;
        let mut channels_first = vec![0.0; ENCODER_DIM * total];
        channels_first[total] = 5.0;
        channels_first[total + 1] = 6.0;

        let (rows, frames) =
            frames_from_channels_first(&[1, ENCODER_DIM, total], &channels_first, &[2])
                .expect("valid encoder output");

        assert_eq!(frames, 2, "only the reported valid frames are kept");
        assert_close(rows[1], 5.0);
        assert_close(rows[ENCODER_DIM + 1], 6.0);
    }

    #[test]
    fn an_encoder_output_of_the_wrong_rank_is_rejected() {
        let err = frames_from_channels_first(&[1, 3], &[0.0; 3], &[1]).expect_err("wrong shape");
        assert!(matches!(err, EngineError::Transcribe(_)));
    }
}
