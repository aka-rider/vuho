//! Voz model loading and per-window inference.
//!
//! Loads the three `CoreML` bundles (mel, encoder, fused decoder) and runs
//! one 15 s window through mel → encoder → greedy TDT decode.

use std::path::Path;

use crate::coreml::{ComputeUnits, CoreMlModel, MlArray, Prediction};
use crate::stream::merge::MergeBounds;
use crate::token::TokenAt;
use crate::vocab::Vocab;
use crate::EngineError;

use super::embedding::Embedding;
use super::frontend;
use super::meta::{feature, Meta};
use super::tdt::{greedy_decode, DecoderStep, LstmState, StepOutput};

/// The decoder's single-lane function.
///
/// The bundle is multifunction — `main` scores 16 lanes at once, `decoder_8`
/// through `decoder_2` in between. Dictation decodes one window at a time,
/// and the 16-lane function would spend fifteen lanes on zeros.
const DECODER_FUNCTION: &str = "decoder_1";

/// Loaded voz models.
pub(crate) struct VozModels {
    mel: CoreMlModel,
    encoder: CoreMlModel,
    decoder: CoreMlModel,
    meta: Meta,
    embedding: Embedding,
    vocab: Vocab,
}

/// One window's encoder output, ready for the decode walk.
struct Encoded {
    /// `enc_frames` frames of `joint_hidden` each, frame after frame.
    projected: Vec<f32>,
    /// How many of those frames carry real audio.
    limit: usize,
}

impl VozModels {
    /// Load all model components for `model_id` from `folder`, then warm
    /// them up ([`Self::warm_up`]).
    ///
    /// Compute units, as the export's author measured them: mel and encoder
    /// on the Neural Engine, whose first plan compilation is slow (~20 s
    /// cold) and then cached; the decoder on CPU, where its small per-step
    /// calls are cheaper than a Neural Engine dispatch.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if the layout is invalid, any
    /// component fails to load, `meta.json` disagrees with the models, the
    /// embedding table, or the vocabulary, or the warm-up inference fails.
    pub(crate) fn load(model_id: &str, folder: &Path) -> Result<Self, EngineError> {
        crate::validate_model_layout(model_id, folder)?;
        let path = |role| crate::asset_path(model_id, role, folder);

        let meta = Meta::load(&path(crate::asset_role::META)?)?;

        log::info!("voz: loading mel and encoder (CPU+ANE)");
        let mel = CoreMlModel::load(
            &path(crate::asset_role::PREPROCESSOR)?,
            ComputeUnits::CpuAndNeuralEngine,
        )?;
        let encoder = CoreMlModel::load(
            &path(crate::asset_role::ENCODER)?,
            ComputeUnits::CpuAndNeuralEngine,
        )?;

        log::info!("voz: loading the decoder (CPU, function {DECODER_FUNCTION})");
        let decoder = CoreMlModel::load_function(
            &path(crate::asset_role::DECODER)?,
            ComputeUnits::CpuOnly,
            DECODER_FUNCTION,
        )?;
        meta.verify_models(&mel, &encoder, &decoder)?;

        let embedding = Embedding::load(&path(crate::asset_role::EMBEDDING)?, &meta)?;
        let vocab = load_vocab(&path(crate::asset_role::VOCAB)?, &meta)?;

        let models = Self {
            mel,
            encoder,
            decoder,
            meta,
            embedding,
            vocab,
        };
        models.warm_up()?;
        Ok(models)
    }

    /// One inference on a zeroed window, to pay `CoreML`'s first-run
    /// compilation before a real session does.
    fn warm_up(&self) -> Result<(), EngineError> {
        log::info!("voz: warmup inference on a zeroed window");
        let started = std::time::Instant::now();
        let zeros = vec![0.0f32; self.meta.n_samples];
        self.decode_window(&zeros, 0)
            .map_err(|e| EngineError::LoadFailed(format!("voz warmup inference failed: {e}")))?;
        log::info!("voz: warmup completed in {:?}", started.elapsed());
        Ok(())
    }

    fn decode_window(
        &self,
        samples: &[f32],
        global_frame_offset: usize,
    ) -> Result<Vec<TokenAt>, EngineError> {
        let encoded = self.encode(samples)?;
        let decoder = FusedDecoder {
            models: self,
            projected: &encoded.projected,
        };
        greedy_decode(&decoder, &self.meta, encoded.limit, global_frame_offset)
    }

    /// Run mel → encoder over `samples` (at most one window) and read the
    /// projected encoder output out of `CoreML`'s buffer.
    fn encode(&self, samples: &[f32]) -> Result<Encoded, EngineError> {
        let meta = &self.meta;
        if samples.len() > meta.n_samples {
            return Err(EngineError::Transcribe(format!(
                "voz takes at most {} samples per window, got {}",
                meta.n_samples,
                samples.len()
            )));
        }
        let mel = self.compute_mel(samples)?;
        let encoded = self.run_encoder(mel, samples.len())?;
        Ok(Encoded {
            projected: read_columns(
                &encoded,
                feature::ENC_PROJ,
                meta.joint_hidden,
                meta.enc_frames,
            )?,
            limit: frontend::valid_encoder_frames(samples.len(), meta),
        })
    }

    fn compute_mel(&self, samples: &[f32]) -> Result<MlArray, EngineError> {
        let meta = &self.meta;
        let audio_rows = MlArray::f16(
            &[1, meta.hop_length, 1, meta.n_rows],
            &frontend::audio_rows(samples, meta),
        )?;
        let mel_mask = MlArray::f16(
            &[1, 1, 1, meta.valid_frames],
            &frontend::mel_mask(samples.len(), meta),
        )?;
        self.mel
            .predict(&[
                (feature::AUDIO_ROWS, audio_rows),
                (feature::MEL_MASK, mel_mask),
            ])?
            .array(feature::MEL)
    }

    fn run_encoder(&self, mel: MlArray, sample_count: usize) -> Result<Prediction, EngineError> {
        let meta = &self.meta;
        let key_bias = MlArray::f16(
            &[1, meta.enc_frames, 1, 1],
            &frontend::key_bias(sample_count, meta),
        )?;
        let pad_mask = MlArray::f16(&[1, 1, 1, meta.enc_frames], &frontend::pad_mask(meta))?;
        self.encoder.predict(&[
            (feature::MEL, mel),
            (feature::KEY_BIAS, key_bias),
            (feature::PAD_MASK, pad_mask),
        ])
    }
}

/// Load the vocabulary and check it holds exactly `vocab_size` pieces —
/// the blank, one past them, is the decoder's alone.
fn load_vocab(path: &Path, meta: &Meta) -> Result<Vocab, EngineError> {
    log::info!("voz: loading vocabulary");
    let vocab = Vocab::load(path, Some(meta.blank_idx))?;
    let last_piece = u32::try_from(meta.vocab_size - 1).ok();
    let holds_exactly = last_piece.is_some_and(|id| vocab.piece_info(id).is_some())
        && vocab.piece_info(meta.blank_idx).is_none();
    if holds_exactly {
        Ok(vocab)
    } else {
        Err(EngineError::LoadFailed(format!(
            "vocab.json does not hold exactly {} pieces",
            meta.vocab_size
        )))
    }
}

/// The element offsets of a `[1, channels, 1, columns]` array's first
/// `columns` columns, column after column, as `CoreML` actually laid it
/// out.
///
/// `CoreML` pads such an array's last dimension (a declared 188 is
/// physically 192), so the offsets come from the array's real strides — a
/// dense read would return misaligned garbage without an error.
fn column_major_offsets(
    strides: &[usize],
    channels: usize,
    columns: usize,
) -> Result<Vec<usize>, EngineError> {
    let &[_, channel_stride, _, column_stride] = strides else {
        return Err(EngineError::CoreMl(format!(
            "expected a 4-D array, CoreML reports strides {strides:?}"
        )));
    };
    Ok((0..columns)
        .flat_map(|column| {
            (0..channels).map(move |ch| ch * channel_stride + column * column_stride)
        })
        .collect())
}

/// Read the first `columns` columns of the `[1, channels, 1, _]` output
/// `name`, column after column (frame-major).
fn read_columns(
    prediction: &Prediction,
    name: &str,
    channels: usize,
    columns: usize,
) -> Result<Vec<f32>, EngineError> {
    let array = prediction.array(name)?;
    let offsets = column_major_offsets(&array.strides(), channels, columns)?;
    let mut values = Vec::new();
    array.gather_f32_into(&offsets, &mut values)?;
    Ok(values)
}

/// `CoreML`-backed [`DecoderStep`]: one fused decoder call per step.
struct FusedDecoder<'a> {
    models: &'a VozModels,
    projected: &'a [f32],
}

impl FusedDecoder<'_> {
    fn predict(
        &self,
        label: u32,
        state: &LstmState,
        position: usize,
        span: usize,
    ) -> Result<StepOutput, EngineError> {
        let meta = &self.models.meta;
        let mut embed = Vec::new();
        self.models.embedding.row_into(label, &mut embed)?;
        let enc_step = frontend::enc_step(self.projected, position, span, meta);

        let recurrent = [1, meta.state_len(), 1, 1];
        let prediction = self.models.decoder.predict(&[
            (
                feature::EMBED,
                MlArray::f16(&[1, meta.pred_hidden, 1, 1], &embed)?,
            ),
            (feature::H_IN, MlArray::f16(&recurrent, &state.h)?),
            (feature::C_IN, MlArray::f16(&recurrent, &state.c)?),
            (
                feature::ENC_STEP,
                MlArray::f16(&[1, meta.joint_hidden, 1, meta.decode_width], &enc_step)?,
            ),
        ])?;

        Ok(StepOutput {
            logits: read_columns(&prediction, feature::LOGITS, meta.n_logits, span)?,
            state: LstmState {
                h: read_columns(&prediction, feature::H_OUT, meta.state_len(), 1)?,
                c: read_columns(&prediction, feature::C_OUT, meta.state_len(), 1)?,
            },
        })
    }
}

impl DecoderStep for FusedDecoder<'_> {
    /// Each call autoreleases IOSurface-backed outputs; draining per step
    /// keeps a long decode from exhausting `IOSurface` allocation (the same
    /// vendor quirk `canary::aed` works around).
    fn step(
        &self,
        label: u32,
        state: &LstmState,
        position: usize,
        span: usize,
    ) -> Result<StepOutput, EngineError> {
        objc2::rc::autoreleasepool(|_| self.predict(label, state, position, span))
    }
}

impl crate::window_inference::WindowInference for VozModels {
    /// Voz decodes without a language prompt, so `language` is unused.
    /// Every window decodes from a fresh state (ADR-015).
    fn infer_window(
        &self,
        samples: &[f32],
        global_frame_offset: usize,
        _language: &str,
    ) -> Result<Vec<TokenAt>, EngineError> {
        self.decode_window(samples, global_frame_offset)
    }

    fn piece_info(&self, id: u32) -> Option<(bool, &str)> {
        self.vocab.piece_info(id)
    }

    fn detokenize(&self, tokens: &[TokenAt]) -> String {
        self.vocab.detokenize(tokens)
    }

    /// Positions are real encoder frame indices.
    fn merge_bounds(&self) -> MergeBounds {
        MergeBounds::measured_positions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A padded `[1, 3, 1, 4]` array declared at 3 columns is physically
    /// 4 wide here: channel stride 4, column stride 1.
    #[test]
    fn offsets_follow_the_reported_strides_not_the_declared_shape() {
        let offsets = column_major_offsets(&[12, 4, 4, 1], 3, 2).expect("4-D strides");
        assert_eq!(offsets, [0, 4, 8, 1, 5, 9]);
    }

    #[test]
    fn a_non_four_dimensional_array_is_an_error() {
        assert!(matches!(
            column_major_offsets(&[4, 1], 3, 2),
            Err(EngineError::CoreMl(_))
        ));
    }
}
