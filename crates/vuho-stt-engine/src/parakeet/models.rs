//! Parakeet-TDT model loading and per-window inference.
//!
//! Loads the four `CoreML` model bundles (Preprocessor, Encoder, Decoder,
//! Joint), runs a full window through the pipeline, and implements
//! [`StepModel`] for the TDT greedy decoder.

use std::path::Path;
use std::sync::Mutex;

use crate::coreml::{ComputeUnits, CoreMlModel, MlArray};
use crate::EngineError;

use super::decoder_state::DecoderState;
use super::tdt::{StepModel, TokenAt};
use super::vocab::Vocab;

/// Encoder output feature dimension.
const ENCODER_DIM: usize = 1024;
/// LSTM hidden/cell state size: `(num_layers=2, batch=1, hidden_dim=640)`.
const H_C_SIZE: usize = 2 * 640;
/// `RNNTJoint` raw logits width: 8193 token logits (8192 vocab + blank) + 5 duration bins.
const JOINT_LOGITS_LEN: usize = 8198;

/// Loaded Parakeet-TDT models.
pub(crate) struct ParakeetModels {
    preprocessor: CoreMlModel,
    encoder: CoreMlModel,
    decoder: CoreMlModel,
    joint: CoreMlModel,
    vocab: Vocab,
    /// Scratch buffer for [`Self::run_encoder`]'s zero-padded 240 000-sample
    /// window (WP9: hot-path allocation discipline) — reused across every
    /// window/partial/commit instead of a fresh `Vec::with_capacity`
    /// allocation per call. `Mutex`, not `RefCell`: `run_encoder` takes
    /// `&self` (this type is only ever reached through
    /// `Arc<SendModel<ParakeetModels>>`, a shared reference across
    /// threads), and while every *current* caller happens to serialize
    /// `CoreML` calls on one thread, nothing in this type structurally
    /// enforces that — `SendModel<ParakeetModels>` is `Sync`, so a second
    /// caller on a second thread (e.g. a future batch `transcribe()` call
    /// racing a live streaming session against the same loaded engine) is
    /// legal Rust as far as the type system is concerned. Two threads
    /// racing a `RefCell`'s non-atomic borrow-flag update is undefined
    /// behavior regardless of whether either racing access happens to
    /// observe the resulting panic — a data race on the flag itself is UB
    /// the instant it occurs, "strictly safer, would just panic" is not a
    /// real property `RefCell` provides under genuine concurrent access,
    /// only under `!Sync` single-threaded re-entrancy. A `Mutex` makes
    /// concurrent `run_encoder` calls block-and-wait instead of racing,
    /// honoring `Sync`'s contract for real; contention is expected to stay
    /// at zero in practice since a lock hold is microseconds and a whole
    /// inference is milliseconds-to-seconds.
    padded: Mutex<Vec<f32>>,
}

impl ParakeetModels {
    /// Load all model components from `folder`.
    ///
    /// Validates the model layout first, then loads each `.mlmodelc`
    /// bundle. Compute-unit choice per component, verified against each
    /// bundle's actual `model.mil` signature (not just the plan's summary
    /// table):
    ///
    /// - `Preprocessor`: CPU-only — it's framing/FFT/mel, all CPU ops, and
    ///   `FluidAudio` does the same.
    /// - `ParakeetEncoder_15s`: CPU + ANE — its inputs are **fixed** shape
    ///   (`[1, 128, 1501]` / `[1]`), so the Neural Engine can compile a
    ///   single plan for it. This is the ANE dispatch that matters.
    /// - `ParakeetDecoder` and `RNNTJoint`: CPU-only. Both declare
    ///   `RangeDims`-flexible inputs (`targets` up to length 1000;
    ///   `encoder_outputs`/`decoder_outputs` up to 1024/1025 respectively).
    ///   Loading either with `CpuAndNeuralEngine` fails at prediction time
    ///   with a `CoreML` E5RT validation error inside the LSTM's internal
    ///   `initial_h` shape check ("expected 1000" — the compiled ANE plan
    ///   locks onto the `RangeDims` ceiling instead of the actual per-step
    ///   shape we run with). These are small per-frame ops anyway, so
    ///   running them on CPU is not a meaningful latency cost.
    ///
    ///   Separately (root-caused, see `CLAUDE.md`'s Known Issues E5RT
    ///   entry): even with `RNNTJoint` correctly loaded `CpuOnly`, merely
    ///   *loading* it (no prediction required) still causes `CoreML`'s E5RT
    ///   runtime to print two harmless "STL exception" lines to stdout
    ///   during process/static teardown — a shape-validation artifact of
    ///   `RNNTJoint.mlmodelc`'s own exported `model.mil`, not anything this
    ///   function or its caller controls. No `Result` this crate observes
    ///   is ever affected by it.
    ///
    /// Warmup runs one full inference on zeros to trigger ANE plan
    /// compilation for the encoder — the first-ever run compiles `CoreML`
    /// for the ANE and can take tens of seconds; see `CLAUDE.md` for
    /// measured cold/warm numbers.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if the model layout is invalid or
    /// any model fails to load.
    pub(crate) fn load(folder: &Path) -> Result<Self, EngineError> {
        crate::validate_model_layout(folder)?;

        log::info!("parakeet: loading Preprocessor (CPU)");
        let preprocessor =
            CoreMlModel::load(&folder.join("Preprocessor.mlmodelc"), ComputeUnits::CpuOnly)?;

        log::info!("parakeet: loading ParakeetEncoder_15s (CPU+ANE)");
        let encoder = CoreMlModel::load(
            &folder.join("ParakeetEncoder_15s.mlmodelc"),
            ComputeUnits::CpuAndNeuralEngine,
        )?;

        log::info!(
            "parakeet: loading ParakeetDecoder (CPU — flexible-shape input, ANE unsupported)"
        );
        let decoder = CoreMlModel::load(
            &folder.join("ParakeetDecoder.mlmodelc"),
            ComputeUnits::CpuOnly,
        )?;

        log::info!("parakeet: loading RNNTJoint (CPU — flexible-shape input, ANE unsupported)");
        let joint = CoreMlModel::load(&folder.join("RNNTJoint.mlmodelc"), ComputeUnits::CpuOnly)?;

        log::info!("parakeet: loading vocabulary");
        let vocab = Vocab::load(&folder.join("parakeet_v3_vocab.json"))?;

        let models = Self {
            preprocessor,
            encoder,
            decoder,
            joint,
            vocab,
            padded: Mutex::new(vec![0.0f32; crate::stream::windower::WINDOW_SAMPLES]),
        };

        log::info!("parakeet: warmup inference on a zeroed window (triggers ANE plan compilation)");
        let started = std::time::Instant::now();
        let zeros = vec![0.0f32; crate::stream::windower::WINDOW_SAMPLES];
        let mut warmup_state = DecoderState::new();
        if let Err(e) = models.infer_window(&zeros, 0, 0, &mut warmup_state) {
            log::warn!("parakeet: warmup inference failed (continuing — a real session will surface the error): {e}");
        }
        log::info!("parakeet: warmup completed in {:?}", started.elapsed());

        Ok(models)
    }

    /// Run a full window through the Parakeet pipeline: Preprocessor →
    /// Encoder → TDT greedy decode.
    ///
    /// `samples` must be ≤ `WINDOW_SAMPLES` (zero-padded internally to the
    /// model's fixed 240 000-sample input). `initial_t` and
    /// `global_frame_offset` are carried from the sliding window state
    /// machine. `state` threads across windows (see [`super::tdt::tdt_greedy`]).
    ///
    /// Returns emitted tokens.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::CoreMl` if a `CoreML` call fails, or if
    /// `RNNTJoint`'s output length doesn't match what the decode loop
    /// expects (a `CoreML`-level output-shape violation, not a failure in
    /// this crate's own algorithm).
    pub(crate) fn infer_window(
        &self,
        samples: &[f32],
        initial_t: usize,
        global_frame_offset: usize,
        state: &mut DecoderState,
    ) -> Result<Vec<TokenAt>, EngineError> {
        let (encoder_output, encoder_frame_count) = self.run_encoder(samples)?;
        let step = StepImpl {
            decoder: &self.decoder,
            joint: &self.joint,
        };
        let (emitted, _next_t) = super::tdt::tdt_greedy(
            &encoder_output,
            encoder_frame_count,
            initial_t,
            global_frame_offset,
            state,
            &step,
        )?;
        // next_t is meaningful only to tdt_greedy's own unit tests
        Ok(emitted)
    }

    /// Detokenize tokens to text using the loaded vocabulary.
    pub(crate) fn detokenize(&self, tokens: &[TokenAt]) -> String {
        self.vocab.detokenize(tokens)
    }

    /// `(is_word_initial, raw_piece_text)` for a token id — see
    /// [`super::vocab::Vocab::piece_info`]. Borrows from `self` (`&str`,
    /// not an owned clone, WP9).
    pub(crate) fn piece_info(&self, id: u32) -> Option<(bool, &str)> {
        self.vocab.piece_info(id)
    }

    /// Run Preprocessor → Encoder, returning `(flat_encoder_output, frame_count)`.
    ///
    /// `flat_encoder_output` is `frame_count` frames of `ENCODER_DIM` each,
    /// row-major, clamped to the model's reported valid length (≤ 188).
    fn run_encoder(&self, samples: &[f32]) -> Result<(Vec<f32>, usize), EngineError> {
        let window_samples = crate::stream::windower::WINDOW_SAMPLES;
        debug_assert!(samples.len() <= window_samples);
        let copy_len = samples.len().min(window_samples);

        // Reuse the scratch buffer across calls (WP9) instead of a fresh
        // `Vec::with_capacity(window_samples)` every window/partial/commit.
        // Every element still gets (re-)written on every call — copy_len
        // bytes from `samples`, the rest zeroed — so a shorter window never
        // observes a longer previous call's stale tail. `unwrap_or_else`
        // recovers from a poisoned lock (rule 12) rather than propagating
        // the panic: the buffer's contents are about to be fully
        // overwritten below regardless of what a panicking prior holder
        // left in it.
        let mut padded = self
            .padded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        padded[..copy_len].copy_from_slice(&samples[..copy_len]);
        padded[copy_len..].fill(0.0);

        let audio_signal = MlArray::f32(&[1, window_samples], &padded)?;
        drop(padded); // release the lock before any other self.padded use.
        let audio_length = MlArray::i32(&[1], &[usize_to_i32(copy_len)])?;

        let prep = self.preprocessor.predict(&[
            ("audio_signal", audio_signal),
            ("audio_length", audio_length),
        ])?;
        // Pass the Preprocessor's own `mel` output straight through to the
        // encoder — no extract-to-Vec-and-rebuild round trip. A rebuilt
        // array is only guaranteed byte-identical if we correctly account
        // for the *model's* returned strides, which a naive contiguous
        // re-serialization does not do; feeding the original `MLMultiArray`
        // straight back in (as CoreML's own APIs are designed for) sidesteps
        // the question entirely.
        let mel = prep.array("mel")?;
        let mel_length = f32_to_usize(prep.array("mel_length")?.to_f32_vec()?[0]);

        let length_arr = MlArray::i32(&[1], &[usize_to_i32(mel_length)])?;
        let enc = self
            .encoder
            .predict(&[("audio_signal", mel), ("length", length_arr)])?;

        let encoder_output = enc.array("encoder_output")?.to_f32_vec()?;
        let encoder_frame_count = encoder_output.len() / ENCODER_DIM;
        let reported_len = f32_to_usize(enc.array("encoder_output_length")?.to_f32_vec()?[0]);
        let valid_frames = reported_len.min(encoder_frame_count);

        log::debug!(
            "parakeet: encoder produced {encoder_frame_count} frames, {valid_frames} valid (reported {reported_len})"
        );

        Ok((encoder_output, valid_frames))
    }
}

/// Convert a small non-negative length/count to `i32` for a `CoreML` scalar
/// input. Every value passed through here is a sample or frame count well
/// under `i32::MAX` (the largest, a window's sample count, is 240 000), so
/// this never truncates or changes sign in practice.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn usize_to_i32(value: usize) -> i32 {
    value as i32
}

/// Convert a `CoreML` scalar length output (returned as `f32`, always a
/// small non-negative integer count in this model set) back to `usize`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn f32_to_usize(value: f32) -> usize {
    value as usize
}

/// `CoreML`-backed [`StepModel`] implementation: one `ParakeetDecoder` call
/// per non-blank emission, one `RNNTJoint` call per encoder frame visited.
struct StepImpl<'a> {
    decoder: &'a CoreMlModel,
    joint: &'a CoreMlModel,
}

impl StepModel for StepImpl<'_> {
    fn decode(&self, token: i32, state: &mut DecoderState) -> Result<(), EngineError> {
        let targets = MlArray::i32(&[1, 1], &[token])?;
        let target_lengths = MlArray::i32(&[1], &[1])?;
        let h_in = MlArray::f32(&[2, 1, 640], &state.h)?;
        let c_in = MlArray::f32(&[2, 1, 640], &state.c)?;

        let prediction = self.decoder.predict(&[
            ("targets", targets),
            ("target_lengths", target_lengths),
            ("h_in", h_in),
            ("c_in", c_in),
        ])?;

        let h_out = prediction.array("h_out")?.to_f32_vec()?;
        let c_out = prediction.array("c_out")?.to_f32_vec()?;
        let dec_out = prediction.array("decoder_output")?.to_f32_vec()?;
        debug_assert_eq!(h_out.len(), H_C_SIZE);
        debug_assert_eq!(c_out.len(), H_C_SIZE);

        state.h = h_out;
        state.c = c_out;
        state.dec_out = Some(dec_out);
        Ok(())
    }

    fn joint(
        &self,
        enc_frame: &[f32],
        dec_out: &[f32],
        out: &mut Vec<f32>,
    ) -> Result<(), EngineError> {
        let enc_outputs = MlArray::f32(&[1, 1, ENCODER_DIM], enc_frame)?;
        let dec_outputs = MlArray::f32(&[1, 1, 640], dec_out)?;

        // RNNTJoint.mlmodelc's real MIL signature (verified against
        // model.mil, not the plan's simplified I/O table) takes exactly
        // these two inputs — no `encoder_length`. Its "logits" output is
        // actually log-softmax, not raw logits, but that's immaterial here:
        // argmax is invariant under the monotonic log-softmax transform.
        let prediction = self.joint.predict(&[
            ("encoder_outputs", enc_outputs),
            ("decoder_outputs", dec_outputs),
        ])?;

        // Write into the caller-owned `out` buffer (WP9) instead of
        // allocating a fresh Vec per frame — `tdt_greedy` reuses `out`
        // across every frame in the decode loop.
        prediction.array("logits")?.to_f32_vec_into(out)?;
        if out.len() != JOINT_LOGITS_LEN {
            return Err(EngineError::CoreMl(format!(
                "RNNTJoint returned {} logits, expected {JOINT_LOGITS_LEN}",
                out.len()
            )));
        }
        Ok(())
    }
}
