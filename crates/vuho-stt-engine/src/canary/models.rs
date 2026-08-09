//! Canary-1B-v2 model loading and per-window inference.
//!
//! Loads the four `CoreML` bundles (Preprocessor, Encoder, Decoder,
//! Projection) and runs one 15 s window through
//! Preprocessor → Encoder → greedy AED decode.

use std::path::Path;
use std::sync::Mutex;

use crate::coreml::{ComputeUnits, CoreMlModel, MlArray};
use crate::stream::merge::MergeBounds;
use crate::token::TokenAt;
use crate::vocab::Vocab;
use crate::EngineError;

use super::aed::ENCODER_FRAMES;
use super::prompt;

/// Slowest per-token rate the position estimate will assume: 4 frames, or
/// 320 ms, per token.
///
/// [`stamp_positions`] spreads a window's tokens over the frames its encoder
/// saw, but only up to this rate. The cap is there because a Canary decode
/// does **not** always reach the end of its window: on `jfk.wav` ×3,
/// window 0's decode covers 13.4 s of its 15 s and stops, so spreading its
/// 43 tokens across all 188 frames puts the last one 1.4 s past the last
/// word it actually transcribed. `stream::merge`'s no-match fallback then
/// drops fresh tokens carrying speech the committed side never had —
/// measured: three words of the second repetition, deleted.
const POS_STRIDE_CEILING: usize = 4;

/// Spread `ids` over the `valid_frames` encoder frames the window covers,
/// starting at `global_frame_offset` and never slower than
/// [`POS_STRIDE_CEILING`].
///
/// Canary's decoder has no acoustic alignment, so a token's position is
/// **estimated**, not measured. The estimate has to land on the same axis
/// Parakeet's measured positions use, because `stream::merge` reads
/// `TokenAt::pos` as an encoder-frame index twice: `search` bounds the seam
/// hunt to `OVERLAP_FRAMES` either side of the last committed position, and
/// the no-match fallback drops every fresh token at or before it.
///
/// A *fixed* stride lands on that axis at exactly one token density (≈47
/// tokens per 188-frame window) and nowhere else. Above it — fast speech —
/// the positions run far past the window they describe and the fallback
/// deletes several times the overlap. Interpolating removes that, and is
/// density-free where it applies: the tokens of a 2 s overlap span ≈25
/// frames whether there are three of them or eleven.
///
/// The two error directions are not equally bad, and that asymmetry is what
/// the cap encodes. Placing a token **too late** deletes speech; placing it
/// **too early** at worst transcribes the overlap twice. The estimate
/// therefore interpolates whenever that puts tokens closer together than the
/// cap, and holds to the cap otherwise — it never claims a token is later
/// than a deliberately slow assumed rate would put it.
fn stamp_positions(ids: &[u32], global_frame_offset: usize, valid_frames: usize) -> Vec<TokenAt> {
    let token_count = ids.len().max(1);
    ids.iter()
        .enumerate()
        .map(|(i, &id)| TokenAt {
            id,
            pos: global_frame_offset + (i * valid_frames / token_count).min(i * POS_STRIDE_CEILING),
        })
        .collect()
}

/// Loaded Canary-1B-v2 models.
pub(crate) struct CanaryModels {
    preprocessor: CoreMlModel,
    encoder: CoreMlModel,
    decoder: CoreMlModel,
    projection: CoreMlModel,
    vocab: Vocab,
    /// The manifest id these models were loaded for — carried so an
    /// unsupported-language error can name the model the user selected
    /// rather than a hardcoded display string.
    model_id: String,
    /// The decoder's declared `input_ids` sequence length, read from the
    /// model: different builds of the same export ship different values.
    decoder_steps: usize,
    /// Scratch buffer for [`Self::run_encoder`]'s zero-padded window,
    /// reused across every window/partial/commit. `Mutex`, not `RefCell`,
    /// for the same reason `ParakeetModels::padded` is (see its field doc):
    /// `SendModel<CanaryModels>` is `Sync`, so concurrent `&self` access is
    /// legal as far as the type system is concerned and must block rather
    /// than race.
    padded: Mutex<Vec<f32>>,
}

/// The four `CoreML` bundles, loaded before anything else needs them.
struct Components {
    preprocessor: CoreMlModel,
    encoder: CoreMlModel,
    decoder: CoreMlModel,
    projection: CoreMlModel,
}

impl Components {
    /// Every component loads CPU-only, unlike Parakeet's encoder.
    ///
    /// This is measured, not assumed. These are `<ios18>` int4 bundles with
    /// fixed shapes, so `CpuAndNeuralEngine` looked like the obvious choice
    /// — but on a cold `CoreML` cache the encoder's ANE plan compilation ran
    /// past **30 minutes** of `ANECompilerService` CPU without finishing,
    /// during which `MLModel` load simply blocks. CPU-only loads all four in
    /// ~4 s and transcribes 11 s of audio in ~0.8 s, which is well inside
    /// the latency this backend needs. The shipped `metadata.json` agrees
    /// (`"compute_units": "CPU_ONLY"`).
    fn load(model_id: &str, folder: &Path) -> Result<Self, EngineError> {
        let path = |role| crate::asset_path(model_id, role, folder);

        log::info!("canary: loading the preprocessor (CPU)");
        let preprocessor = CoreMlModel::load(
            &path(crate::asset_role::PREPROCESSOR)?,
            ComputeUnits::CpuOnly,
        )?;

        log::info!("canary: loading the encoder (CPU)");
        let encoder = CoreMlModel::load(&path(crate::asset_role::ENCODER)?, ComputeUnits::CpuOnly)?;

        log::info!("canary: loading the decoder (CPU)");
        let decoder = CoreMlModel::load(&path(crate::asset_role::DECODER)?, ComputeUnits::CpuOnly)?;

        log::info!("canary: loading the projection (CPU)");
        let projection =
            CoreMlModel::load(&path(crate::asset_role::PROJECTION)?, ComputeUnits::CpuOnly)?;

        Ok(Self {
            preprocessor,
            encoder,
            decoder,
            projection,
        })
    }
}

impl CanaryModels {
    /// Load all four model components for `model_id` from `folder`, then
    /// warm them up ([`Self::warm_up`]).
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if the layout is invalid, any
    /// component fails to load, or the decoder does not declare an
    /// `input_ids` sequence length.
    pub(crate) fn load(model_id: &str, folder: &Path) -> Result<Self, EngineError> {
        crate::validate_model_layout(model_id, folder)?;
        let components = Components::load(model_id, folder)?;

        let decoder_steps = decoder_steps(&components.decoder)?;
        log::info!("canary: decoder sequence length is {decoder_steps}");

        log::info!("canary: loading vocabulary");
        let vocab = Vocab::load(
            &crate::asset_path(model_id, crate::asset_role::VOCAB, folder)?,
            None,
        )?;

        let models = Self {
            preprocessor: components.preprocessor,
            encoder: components.encoder,
            decoder: components.decoder,
            projection: components.projection,
            vocab,
            model_id: model_id.to_owned(),
            decoder_steps,
            padded: Mutex::new(vec![0.0f32; crate::stream::windower::WINDOW_SAMPLES]),
        };
        models.warm_up();
        Ok(models)
    }

    /// One inference on a zeroed window, to pay `CoreML`'s first-run
    /// compilation before a real session does. A failure is logged, not
    /// fatal — a real session surfaces the same error with real audio
    /// behind it.
    fn warm_up(&self) {
        log::info!("canary: warmup inference on a zeroed window");
        let started = std::time::Instant::now();
        let zeros = vec![0.0f32; crate::stream::windower::WINDOW_SAMPLES];
        if let Err(e) = self.decode_window(&zeros, prompt::WARMUP_LANGUAGE) {
            log::warn!("canary: warmup inference failed (continuing — a real session will surface the error): {e}");
        }
        log::info!("canary: warmup completed in {:?}", started.elapsed());
    }

    /// Run one window end to end and return the emitted token ids together
    /// with the number of encoder frames that carried real audio — the span
    /// [`stamp_positions`] spreads those ids over.
    fn decode_window(
        &self,
        samples: &[f32],
        language: &str,
    ) -> Result<(Vec<u32>, usize), EngineError> {
        let prompt = prompt::transcribe_prompt(language).ok_or_else(|| {
            EngineError::UnsupportedLanguage {
                model: self.display_name(),
                language: language.to_owned(),
            }
        })?;
        let (encoder_embeddings, valid_frames) = self.run_encoder(samples)?;
        let ids = super::aed::greedy_decode(
            &self.decoder,
            &self.projection,
            &encoder_embeddings,
            valid_frames,
            self.decoder_steps,
            &prompt,
        )?;
        Ok((ids, valid_frames))
    }

    /// The manifest's display name for this model, for user-facing errors.
    fn display_name(&self) -> String {
        vuho_model_paths::manifest()
            .stt
            .model(&self.model_id)
            .map_or_else(|| self.model_id.clone(), |m| m.display_name.clone())
    }

    /// Build the preprocessor's two inputs for `samples`.
    ///
    /// The preprocessor's input is a FIXED `[1, 240 000]`, but the streaming
    /// path calls this with an open buffer that starts around a second and
    /// grows — so the window is zero-padded into a reused scratch buffer and
    /// the **unpadded** count is passed separately as `audio_length`.
    fn window_inputs(&self, samples: &[f32]) -> Result<(MlArray, MlArray), EngineError> {
        let window_samples = crate::stream::windower::WINDOW_SAMPLES;
        debug_assert!(samples.len() <= window_samples);
        let copy_len = samples.len().min(window_samples);

        // `unwrap_or_else` recovers from a poisoned lock (rule 12) — every
        // element is overwritten below regardless of what a panicking prior
        // holder left behind.
        let mut padded = self
            .padded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        padded[..copy_len].copy_from_slice(&samples[..copy_len]);
        padded[copy_len..].fill(0.0);

        let audio_signal = MlArray::f32(&[1, window_samples], &padded)?;
        drop(padded); // release the lock before any other self.padded use.
        Ok((audio_signal, MlArray::i32(&[1], &[usize_to_i32(copy_len)])?))
    }

    /// Run Preprocessor → Encoder, returning the encoder output already
    /// transposed to `[ENCODER_FRAMES][ENCODER_DIM]` row-major (what the
    /// decoder's `encoder_embeddings` input takes) plus the number of frames
    /// that carry real audio.
    fn run_encoder(&self, samples: &[f32]) -> Result<(Vec<f32>, usize), EngineError> {
        let (audio_signal, audio_length) = self.window_inputs(samples)?;
        let prep = self.preprocessor.predict(&[
            ("audio_signal", audio_signal),
            ("audio_length", audio_length),
        ])?;
        // Pass the preprocessor's own outputs straight through to the
        // encoder — no extract-to-Vec-and-rebuild round trip (ADR-014).
        let enc = self.encoder.predict(&[
            ("features", prep.array("processed")?),
            ("features_length", prep.array("processed_length")?),
        ])?;

        let encoder_out = enc.array("encoder")?;
        let offsets = super::aed::transposed_encoder_offsets(&encoder_out.strides())?;
        let mut embeddings = Vec::with_capacity(offsets.len());
        encoder_out.gather_f32_into(&offsets, &mut embeddings)?;

        let reported = f32_to_usize(enc.array("encoder_length")?.to_f32_vec()?[0]);
        let valid_frames = reported.min(ENCODER_FRAMES);
        log::debug!("canary: encoder reported {reported} frames, {valid_frames} valid");

        Ok((embeddings, valid_frames))
    }
}

/// Read the decoder's declared `input_ids` sequence length from the model
/// itself — different builds of this export ship different values, so
/// hardcoding today's would silently mis-shape the token tensor on another.
fn decoder_steps(decoder: &CoreMlModel) -> Result<usize, EngineError> {
    decoder
        .input_shape("input_ids")
        .and_then(|shape| shape.last().copied())
        .ok_or_else(|| {
            EngineError::LoadFailed(
                "the Canary decoder declares no input_ids sequence length".to_owned(),
            )
        })
}

impl crate::window_inference::WindowInference for CanaryModels {
    /// Decode one window, stamping each token with an estimated encoder-frame
    /// position (see [`stamp_positions`]).
    fn infer_window(
        &self,
        samples: &[f32],
        global_frame_offset: usize,
        language: &str,
    ) -> Result<Vec<TokenAt>, EngineError> {
        let (ids, valid_frames) = self.decode_window(samples, language)?;
        Ok(stamp_positions(&ids, global_frame_offset, valid_frames))
    }

    fn piece_info(&self, id: u32) -> Option<(bool, &str)> {
        self.vocab.piece_info(id)
    }

    fn detokenize(&self, tokens: &[TokenAt]) -> String {
        self.vocab.detokenize(tokens)
    }

    /// Unbounded `tolerance`, bounded `search` — the two halves of
    /// `MergeBounds` pull in opposite directions here, and only `tolerance`
    /// may be widened.
    ///
    /// An interpolated position ([`stamp_positions`]) locates the overlap
    /// *region* reliably but a single word only to within however far that
    /// window's speech departs from an even rate — seconds, on audio that
    /// starts or ends in silence. Requiring two otherwise-identical words to
    /// sit within a few frames of each other would therefore reject real
    /// seam matches: `tolerance` goes away entirely.
    ///
    /// `search` must stay bounded to the physical overlap regardless.
    /// Widening it to `usize::MAX` makes the matcher consider the whole
    /// transcript, and on genuinely repetitive speech it then prefers a
    /// long, spurious match far from the seam over the short true one —
    /// measured, not hypothesized: a buffer holding the same utterance three
    /// times came back with it twice, one repetition truncated away by the
    /// re-splice.
    fn merge_bounds(&self) -> MergeBounds {
        MergeBounds {
            search: crate::stream::windower::OVERLAP_FRAMES,
            tolerance: usize::MAX,
        }
    }
}

/// Convert a small non-negative sample count to the `i32` a `CoreML` scalar
/// input takes. The largest value here is a window's 240 000 samples.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn usize_to_i32(value: usize) -> i32 {
    value as i32
}

/// Convert a `CoreML` scalar length output (always a small non-negative
/// integer count in this model set) back to `usize`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn f32_to_usize(value: f32) -> usize {
    value as usize
}

/// How [`stamp_positions`] behaves once `stream::merge` reads the result —
/// the composition is the contract, so these drive the real `merge` with
/// synthetic token streams rather than asserting on positions alone. No
/// model is involved: both halves are pure.
#[cfg(test)]
mod tests {
    use super::super::aed::ENCODER_FRAMES;
    use super::*;
    use crate::stream::merge::merge;
    use crate::stream::windower::{ADVANCE, OVERLAP_SAMPLES, SAMPLES_PER_FRAME, WINDOW_SAMPLES};

    /// The token densities a 15 s window really spans: a slow speaker, the
    /// ≈47-tokens-per-window rate the retired fixed stride was scaled to,
    /// and a fast one.
    const DENSITIES: [u32; 3] = [20, 47, 80];

    /// Every id is its own whole word; the text is `w{id}` so two ids never
    /// compare equal, and id `0` resolves to `None` the way a blank or
    /// unknown id does. Leaks, like `stream::merge`'s own test vocabulary
    /// does, to satisfy the borrowing signature — a few hundred short
    /// strings per run, never a hot path.
    fn word_piece(id: u32) -> Option<(bool, &'static str)> {
        if id == 0 {
            return None;
        }
        Some((true, Box::leak(format!(" w{id}").into_boxed_str())))
    }

    /// The bounds `CanaryModels::merge_bounds` supplies, without a model.
    fn bounds() -> crate::stream::merge::MergeBounds {
        crate::stream::merge::MergeBounds {
            search: crate::stream::windower::OVERLAP_FRAMES,
            tolerance: usize::MAX,
        }
    }

    /// How many of a window's `tokens_per_window` words fall inside the 2 s
    /// overlap the next window re-decodes.
    fn overlap_tokens(tokens_per_window: u32) -> u32 {
        let overlap = u32::try_from(OVERLAP_SAMPLES).expect("32 000 fits in a u32");
        let window = u32::try_from(WINDOW_SAMPLES).expect("240 000 fits in a u32");
        tokens_per_window * overlap / window
    }

    /// `count` as the token count `merge` reports in `usize`.
    fn token_count(count: usize) -> u32 {
        u32::try_from(count).expect("these tests use fewer than u32::MAX tokens")
    }

    /// The second window's position offset: one `ADVANCE` on from the first.
    fn second_window_offset() -> usize {
        ADVANCE / SAMPLES_PER_FRAME
    }

    fn ids(tokens: &[TokenAt]) -> Vec<u32> {
        tokens.iter().map(|t| t.id).collect()
    }

    /// Two consecutive full windows whose decodes agree about the overlap:
    /// the second window re-decodes the first's trailing overlap words
    /// identically, then continues. The merged transcript must be the union,
    /// with the shared words appearing exactly once.
    #[test]
    fn an_agreed_seam_is_spliced_once_at_every_density() {
        for tokens_per_window in DENSITIES {
            let shared = overlap_tokens(tokens_per_window);
            let first: Vec<u32> = (1..=tokens_per_window).collect();
            let second: Vec<u32> =
                (1 + tokens_per_window - shared..=2 * tokens_per_window - shared).collect();

            let committed = stamp_positions(&first, 0, ENCODER_FRAMES);
            let fresh = stamp_positions(&second, second_window_offset(), ENCODER_FRAMES);

            let outcome = merge(&committed, fresh, bounds(), word_piece);
            let mut merged = ids(&committed[..outcome.keep_committed]);
            merged.extend(ids(&outcome.append));

            let expected: Vec<u32> = (1..=2 * tokens_per_window - shared).collect();
            assert_eq!(
                merged, expected,
                "{tokens_per_window} tokens/window: the {shared} overlap words must survive exactly once"
            );
        }
    }

    /// How many fresh tokens `merge`'s no-match fallback drops when the two
    /// windows' decodes share no word at all.
    fn fallback_drop_count(tokens_per_window: u32) -> u32 {
        let first: Vec<u32> = (1..=tokens_per_window).collect();
        let second: Vec<u32> = (1000..1000 + tokens_per_window).collect();

        let committed = stamp_positions(&first, 0, ENCODER_FRAMES);
        let fresh = stamp_positions(&second, second_window_offset(), ENCODER_FRAMES);

        let outcome = merge(&committed, fresh, bounds(), word_piece);
        assert_eq!(
            outcome.keep_committed,
            committed.len(),
            "{tokens_per_window} tokens/window: an unmatched seam must not truncate committed"
        );
        tokens_per_window - token_count(outcome.append.len())
    }

    /// When the two decodes disagree about every overlap word, `merge` falls
    /// back to dropping fresh tokens by position — and that fallback must
    /// never drop more than the overlap holds. Dropping more deletes speech
    /// the first window never transcribed, which is the failure this whole
    /// estimate is shaped to avoid.
    #[test]
    fn the_no_match_fallback_never_drops_more_than_the_overlap() {
        for tokens_per_window in DENSITIES {
            let shared = overlap_tokens(tokens_per_window);
            let dropped = fallback_drop_count(tokens_per_window);
            assert!(
                dropped <= shared + 1,
                "{tokens_per_window} tokens/window: dropped {dropped} fresh tokens, more than the {shared} the overlap holds"
            );
        }
    }

    /// Known limit, pinned so it stays a decision rather than a surprise
    /// (ADR-022). A sparse window's positions are held to
    /// [`POS_STRIDE_CEILING`] and so stop well short of the window's end, and
    /// the fallback keys off the last committed position — so a sparse
    /// window whose two decodes *also* share no word drops nothing and
    /// transcribes the overlap twice. Duplication is the accepted direction:
    /// the estimate that would drop here is the one that deletes speech
    /// elsewhere.
    #[test]
    fn a_sparse_window_with_no_shared_word_still_duplicates_its_overlap() {
        assert_eq!(fallback_drop_count(20), 0);
    }

    /// Positions stay non-decreasing within a window and never run past the
    /// window's own frame span — what makes them comparable with Parakeet's
    /// measured ones, and what the seam search bound relies on.
    #[test]
    fn positions_stay_inside_the_window_frame_span() {
        for tokens_per_window in DENSITIES {
            let all: Vec<u32> = (1..=tokens_per_window).collect();
            let stamped = stamp_positions(&all, 100, ENCODER_FRAMES);
            assert!(stamped.windows(2).all(|p| p[0].pos <= p[1].pos));
            assert_eq!(stamped[0].pos, 100);
            assert!(stamped.last().unwrap().pos < 100 + ENCODER_FRAMES);
        }
    }

    #[test]
    fn stamping_no_tokens_yields_no_tokens() {
        assert!(stamp_positions(&[], 0, ENCODER_FRAMES).is_empty());
    }
}
