//! Canary's greedy attention-encoder-decoder loop.
//!
//! One decoder prediction plus one projection prediction per emitted
//! token. There is **no KV cache** in this export: the whole `[1, S]` token
//! tensor is resubmitted every step, with `decoder_mask` marking how much of
//! it is real. The hidden state to project is row `pos - 1` of the decoder's
//! `[1, S, 1024]` output — the position that just attended to everything
//! written so far.

use crate::coreml::{CoreMlModel, MlArray};
use crate::EngineError;

use super::prompt::{EOS_ID, PROMPT_LEN};

/// Encoder output frames per 15 s window (fixed by the export).
pub(crate) const ENCODER_FRAMES: usize = 188;
/// Encoder/decoder hidden width.
pub(crate) const ENCODER_DIM: usize = 1024;
/// Vocabulary size — the width of `Projection`'s `logits` output.
const VOCAB_LEN: usize = 16384;

/// Everything a decode step needs that does not change between steps.
struct Context<'a> {
    decoder: &'a CoreMlModel,
    projection: &'a CoreMlModel,
    encoder_embeddings: MlArray,
    encoder_mask: MlArray,
    steps: usize,
}

impl<'a> Context<'a> {
    fn new(
        decoder: &'a CoreMlModel,
        projection: &'a CoreMlModel,
        encoder_embeddings: &[f32],
        valid_frames: usize,
        steps: usize,
    ) -> Result<Self, EngineError> {
        let mask: Vec<f32> = (0..ENCODER_FRAMES)
            .map(|i| if i < valid_frames { 1.0 } else { 0.0 })
            .collect();
        Ok(Self {
            decoder,
            projection,
            encoder_embeddings: MlArray::f32(
                &[1, ENCODER_FRAMES, ENCODER_DIM],
                encoder_embeddings,
            )?,
            encoder_mask: MlArray::f32(&[1, ENCODER_FRAMES], &mask)?,
            steps,
        })
    }
}

/// The running decode: the `[1, S]` token tensor and its mask, plus the
/// scratch buffers reused across steps.
struct DecodeState {
    input_ids: Vec<i32>,
    decoder_mask: Vec<f32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    row_offsets: Vec<usize>,
}

impl DecodeState {
    /// The token tensor is **zero**-filled, not `pad_id`-filled: the
    /// reference implementation pads `input_ids` with `<unk>` and relies on
    /// `decoder_mask` alone to mark what is real. Counter-intuitive, and
    /// load-bearing.
    fn seeded(steps: usize, prompt: &[i32; PROMPT_LEN]) -> Self {
        let mut state = Self {
            input_ids: vec![0; steps],
            decoder_mask: vec![0.0; steps],
            hidden: Vec::with_capacity(ENCODER_DIM),
            logits: Vec::with_capacity(VOCAB_LEN),
            row_offsets: Vec::with_capacity(ENCODER_DIM),
        };
        state.input_ids[..PROMPT_LEN].copy_from_slice(prompt);
        state.decoder_mask[..PROMPT_LEN].fill(1.0);
        state
    }
}

/// Greedily decode one encoded window.
///
/// `encoder_embeddings` is `[ENCODER_FRAMES][ENCODER_DIM]` row-major (the
/// transposed encoder output); `valid_frames` is how many of those frames
/// carry real audio. `steps` is the decoder's declared sequence length,
/// read from the model rather than assumed.
///
/// Returns the emitted token ids with the prompt stripped and `EOS_ID` not
/// included.
///
/// # Errors
///
/// Returns `EngineError::CoreMl` if a `CoreML` call fails or an output has
/// an unexpected length.
pub(crate) fn greedy_decode(
    decoder: &CoreMlModel,
    projection: &CoreMlModel,
    encoder_embeddings: &[f32],
    valid_frames: usize,
    steps: usize,
    prompt: &[i32; PROMPT_LEN],
) -> Result<Vec<u32>, EngineError> {
    if steps <= PROMPT_LEN {
        return Err(EngineError::CoreMl(format!(
            "decoder sequence length {steps} leaves no room for the {PROMPT_LEN}-token prompt"
        )));
    }

    let context = Context::new(decoder, projection, encoder_embeddings, valid_frames, steps)?;
    let mut state = DecodeState::seeded(steps, prompt);

    let mut emitted = Vec::new();
    for pos in PROMPT_LEN..steps {
        // Each step autoreleases MB-scale IOSurface-backed CoreML outputs;
        // draining per step is what keeps a long decode from exhausting
        // IOSurface allocation (vendor quirk — FluidAudio hit exactly this).
        let next = objc2::rc::autoreleasepool(|_| step(&context, &mut state, pos))?;
        if next == EOS_ID {
            break;
        }
        emitted.push(next);
        state.input_ids[pos] = id_to_i32(next);
        state.decoder_mask[pos] = 1.0;
    }
    Ok(emitted)
}

/// One decode step: predict the decoder over the whole token tensor, read
/// the hidden state at row `pos - 1`, project it, and take the argmax.
fn step(context: &Context<'_>, state: &mut DecodeState, pos: usize) -> Result<u32, EngineError> {
    let shape = [1, context.steps];
    let prediction = context.decoder.predict(&[
        ("input_ids", MlArray::i32(&shape, &state.input_ids)?),
        ("decoder_mask", MlArray::f32(&shape, &state.decoder_mask)?),
        ("encoder_embeddings", context.encoder_embeddings.clone()),
        ("encoder_mask", context.encoder_mask.clone()),
    ])?;

    read_hidden_row(&prediction.array("decoder")?, pos - 1, state)?;

    let projected = context
        .projection
        .predict(&[("hidden", MlArray::f32(&[1, ENCODER_DIM], &state.hidden)?)])?;
    projected
        .array("logits")?
        .to_f32_vec_into(&mut state.logits)?;
    if state.logits.len() != VOCAB_LEN {
        return Err(EngineError::CoreMl(format!(
            "Projection returned {} logits, expected {VOCAB_LEN}",
            state.logits.len()
        )));
    }
    Ok(argmax(&state.logits))
}

/// Read row `row` of a `[1, S, ENCODER_DIM]` decoder output into
/// `state.hidden`, honoring the array's real strides — `CoreML` pads an
/// array's last dimension to a 64-element boundary, so a dense read is not
/// guaranteed to land on the right elements.
fn read_hidden_row(
    decoder_out: &MlArray,
    row: usize,
    state: &mut DecodeState,
) -> Result<(), EngineError> {
    let strides = decoder_out.strides();
    let [_, row_stride, column_stride] = strides[..] else {
        return Err(EngineError::CoreMl(format!(
            "decoder output has {} dimensions, expected 3",
            strides.len()
        )));
    };
    let base = row * row_stride;
    state.row_offsets.clear();
    state
        .row_offsets
        .extend((0..ENCODER_DIM).map(|c| base + c * column_stride));
    decoder_out.gather_f32_into(&state.row_offsets, &mut state.hidden)
}

/// Element offsets that read a `[1, ENCODER_DIM, ENCODER_FRAMES]`
/// channels-first encoder output out in `[ENCODER_FRAMES, ENCODER_DIM]`
/// row-major order — the transpose the decoder's `encoder_embeddings` input
/// wants, expressed as a gather so the array's real (padded) strides are
/// honored rather than assumed dense.
///
/// # Errors
///
/// Returns `EngineError::CoreMl` if the encoder output is not 3-dimensional.
pub(crate) fn transposed_encoder_offsets(strides: &[usize]) -> Result<Vec<usize>, EngineError> {
    let [_, channel_stride, frame_stride] = strides[..] else {
        return Err(EngineError::CoreMl(format!(
            "encoder output has {} dimensions, expected 3",
            strides.len()
        )));
    };
    let mut offsets = Vec::with_capacity(ENCODER_FRAMES * ENCODER_DIM);
    for frame in 0..ENCODER_FRAMES {
        offsets.extend((0..ENCODER_DIM).map(|c| c * channel_stride + frame * frame_stride));
    }
    Ok(offsets)
}

/// Index of the largest logit. Ties go to the lowest id, matching a
/// reference greedy decode.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    // The vocabulary is 16384 entries, far inside u32.
    #[allow(clippy::cast_possible_truncation)]
    let id = best as u32;
    id
}

/// Widen a vocabulary id to the `i32` the decoder's `input_ids` takes. The
/// vocabulary is 16384 entries, so this never wraps.
#[allow(clippy::cast_possible_wrap)]
fn id_to_i32(id: u32) -> i32 {
    id as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_the_largest_and_breaks_ties_low() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax(&[0.9, 0.9]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY, -1.0]), 1);
    }

    /// The transpose must reorder a channels-first encoder output into
    /// frame-major order, and must read through the padded stride rather
    /// than the dense one (the 188 → 192 padding is exactly what a dense
    /// read would silently get wrong).
    #[test]
    fn transposed_offsets_honor_a_padded_last_dimension() {
        let padded_frame_stride = 192;
        let offsets = transposed_encoder_offsets(&[
            ENCODER_DIM * padded_frame_stride,
            padded_frame_stride,
            1,
        ])
        .expect("3-d strides");

        assert_eq!(offsets.len(), ENCODER_FRAMES * ENCODER_DIM);
        // Destination [frame 0][channel 0] reads source channel 0, frame 0.
        assert_eq!(offsets[0], 0);
        // Destination [frame 0][channel 1] reads source channel 1, frame 0 —
        // one padded channel stride away, not one element.
        assert_eq!(offsets[1], padded_frame_stride);
        // Destination [frame 1][channel 0] reads source channel 0, frame 1.
        assert_eq!(offsets[ENCODER_DIM], 1);
    }

    #[test]
    fn a_non_three_dimensional_encoder_output_is_an_error() {
        assert!(transposed_encoder_offsets(&[1, 1]).is_err());
    }
}
