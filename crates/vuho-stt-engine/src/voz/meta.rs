//! The typed `meta.json` shipped beside the voz bundles — the one source
//! of every geometry number this backend uses — and its cross-check
//! against what the loaded models themselves report.

use std::path::Path;

use serde::Deserialize;

use vuho_audio::OUTPUT_SAMPLE_RATE;

use crate::coreml::CoreMlModel;
use crate::stream::windower::{SAMPLES_PER_FRAME, WINDOW_SAMPLES};
use crate::EngineError;

/// Mel frames the encoder folds into one encoder frame.
pub(crate) const ENCODER_SUBSAMPLING: usize = 8;

/// The models' input and output feature names.
pub(crate) mod feature {
    pub(crate) const AUDIO_ROWS: &str = "audio_rows";
    pub(crate) const MEL_MASK: &str = "mel_mask";
    pub(crate) const MEL: &str = "mel";
    pub(crate) const KEY_BIAS: &str = "key_bias";
    pub(crate) const PAD_MASK: &str = "pad_mask";
    pub(crate) const ENC_PROJ: &str = "enc_proj";
    pub(crate) const EMBED: &str = "embed";
    pub(crate) const H_IN: &str = "h_in";
    pub(crate) const C_IN: &str = "c_in";
    pub(crate) const ENC_STEP: &str = "enc_step";
    pub(crate) const LOGITS: &str = "logits";
    pub(crate) const H_OUT: &str = "h_out";
    pub(crate) const C_OUT: &str = "c_out";
}

/// The subset of `meta.json` this backend reads.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Meta {
    pub(crate) sample_rate: u32,
    pub(crate) io_precision: String,
    pub(crate) n_samples: usize,
    pub(crate) n_padded_samples: usize,
    pub(crate) n_fft: usize,
    pub(crate) hop_length: usize,
    pub(crate) preemph: f32,
    pub(crate) n_rows: usize,
    pub(crate) n_mels: usize,
    pub(crate) valid_frames: usize,
    pub(crate) enc_frames: usize,
    pub(crate) pred_hidden: usize,
    pub(crate) pred_layers: usize,
    pub(crate) joint_hidden: usize,
    pub(crate) decode_width: usize,
    pub(crate) vocab_size: usize,
    pub(crate) blank_idx: u32,
    pub(crate) n_logits: usize,
    pub(crate) num_durations: usize,
    pub(crate) durations: Vec<usize>,
}

impl Meta {
    /// Read and validate `path`.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if the file cannot be read or
    /// parsed, or its numbers contradict each other or this crate's
    /// window geometry.
    pub(crate) fn load(path: &Path) -> Result<Self, EngineError> {
        let data = std::fs::read_to_string(path).map_err(|e| {
            EngineError::LoadFailed(format!("failed to read {}: {e}", path.display()))
        })?;
        let meta: Self = serde_json::from_str(&data).map_err(|e| {
            EngineError::LoadFailed(format!("failed to parse {}: {e}", path.display()))
        })?;
        meta.validate()?;
        Ok(meta)
    }

    /// Length of the LSTM's `h` (and of its `c`): every layer stacked.
    pub(crate) fn state_len(&self) -> usize {
        self.pred_layers * self.pred_hidden
    }

    fn validate(&self) -> Result<(), EngineError> {
        let failed = self
            .window_checks()
            .into_iter()
            .chain(self.vocabulary_checks())
            .find(|(holds, _)| !holds);
        match failed {
            Some((_, why)) => Err(EngineError::LoadFailed(format!("voz meta.json: {why}"))),
            None => Ok(()),
        }
    }

    fn window_checks(&self) -> [(bool, &'static str); 8] {
        [
            (
                self.sample_rate == OUTPUT_SAMPLE_RATE,
                "sample_rate is not the pipeline's capture rate",
            ),
            (
                self.io_precision == "float16",
                "io_precision is not float16",
            ),
            (
                self.n_samples == WINDOW_SAMPLES,
                "n_samples is not the pipeline's 15 s window",
            ),
            (
                self.hop_length * ENCODER_SUBSAMPLING == SAMPLES_PER_FRAME,
                "an encoder frame is not the pipeline's 1280 samples",
            ),
            (
                self.n_padded_samples == self.n_rows * self.hop_length,
                "n_padded_samples is not n_rows * hop_length",
            ),
            (
                self.n_fft / 2 + self.n_samples <= self.n_padded_samples,
                "the left pad and window do not fit in n_padded_samples",
            ),
            (
                self.valid_frames <= self.n_rows,
                "valid_frames exceeds n_rows",
            ),
            (
                self.valid_frames.div_ceil(ENCODER_SUBSAMPLING) <= self.enc_frames,
                "enc_frames cannot hold valid_frames after subsampling",
            ),
        ]
    }

    fn vocabulary_checks(&self) -> [(bool, &'static str); 4] {
        [
            (
                self.blank_idx as usize == self.vocab_size,
                "blank_idx is not vocab_size",
            ),
            (
                self.n_logits == self.vocab_size + 1 + self.num_durations,
                "n_logits is not vocab + blank + durations",
            ),
            (
                self.durations.len() == self.num_durations,
                "durations does not have num_durations entries",
            ),
            (self.decode_width > 0, "decode_width is zero"),
        ]
    }

    /// Check every tensor shape this backend feeds or reads against what
    /// the loaded models declare.
    ///
    /// `meta.json` and the `.mlmodelc` bundles ship as separate files; if
    /// they ever disagree, every window would fail (or, worse, decode
    /// garbage) — so the disagreement is a load failure naming the tensor.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` for the first tensor whose declared
    /// shape differs from `meta.json`'s.
    pub(crate) fn verify_models(
        &self,
        mel: &CoreMlModel,
        encoder: &CoreMlModel,
        decoder: &CoreMlModel,
    ) -> Result<(), EngineError> {
        let expectations = [
            self.mel_expectations(mel),
            self.encoder_expectations(encoder),
            self.decoder_expectations(decoder),
        ];
        for expectation in expectations.iter().flatten() {
            expectation.verify()?;
        }
        Ok(())
    }

    fn mel_expectations<'a>(&self, mel: &'a CoreMlModel) -> Vec<Expectation<'a>> {
        let (rows, valid) = (self.n_rows, self.valid_frames);
        vec![
            Expectation::input(
                "mel",
                mel,
                feature::AUDIO_ROWS,
                [1, self.hop_length, 1, rows],
            ),
            Expectation::input("mel", mel, feature::MEL_MASK, [1, 1, 1, valid]),
            Expectation::output("mel", mel, feature::MEL, [1, self.n_mels, 1, valid]),
        ]
    }

    fn encoder_expectations<'a>(&self, encoder: &'a CoreMlModel) -> Vec<Expectation<'a>> {
        let (mels, enc) = (self.n_mels, self.enc_frames);
        vec![
            Expectation::input(
                "encoder",
                encoder,
                feature::MEL,
                [1, mels, 1, self.valid_frames],
            ),
            Expectation::input("encoder", encoder, feature::KEY_BIAS, [1, enc, 1, 1]),
            Expectation::input("encoder", encoder, feature::PAD_MASK, [1, 1, 1, enc]),
            Expectation::output(
                "encoder",
                encoder,
                feature::ENC_PROJ,
                [1, self.joint_hidden, 1, enc],
            ),
        ]
    }

    fn decoder_expectations<'a>(&self, decoder: &'a CoreMlModel) -> Vec<Expectation<'a>> {
        let state = [1, self.state_len(), 1, 1];
        let (joint, width) = (self.joint_hidden, self.decode_width);
        vec![
            Expectation::input(
                "decoder",
                decoder,
                feature::EMBED,
                [1, self.pred_hidden, 1, 1],
            ),
            Expectation::input("decoder", decoder, feature::H_IN, state),
            Expectation::input("decoder", decoder, feature::C_IN, state),
            Expectation::input("decoder", decoder, feature::ENC_STEP, [1, joint, 1, width]),
            Expectation::output(
                "decoder",
                decoder,
                feature::LOGITS,
                [1, self.n_logits, 1, width],
            ),
            Expectation::output("decoder", decoder, feature::H_OUT, state),
            Expectation::output("decoder", decoder, feature::C_OUT, state),
        ]
    }
}

/// One tensor's shape as `meta.json` implies it.
struct Expectation<'a> {
    component: &'static str,
    name: &'static str,
    side: Side<'a>,
    shape: [usize; 4],
}

impl<'a> Expectation<'a> {
    fn input(
        component: &'static str,
        model: &'a CoreMlModel,
        name: &'static str,
        shape: [usize; 4],
    ) -> Self {
        Self {
            component,
            name,
            side: Side::In(model),
            shape,
        }
    }

    fn output(
        component: &'static str,
        model: &'a CoreMlModel,
        name: &'static str,
        shape: [usize; 4],
    ) -> Self {
        Self {
            component,
            name,
            side: Side::Out(model),
            shape,
        }
    }

    fn verify(&self) -> Result<(), EngineError> {
        let declared = self.side.declared_shape(self.name);
        if declared.as_deref() == Some(self.shape.as_slice()) {
            return Ok(());
        }
        Err(EngineError::LoadFailed(format!(
            "voz {} {} '{}' declares {declared:?}, meta.json implies {:?}",
            self.component,
            self.side.label(),
            self.name,
            self.shape
        )))
    }
}

/// Which side of a model a feature name lives on.
enum Side<'a> {
    In(&'a CoreMlModel),
    Out(&'a CoreMlModel),
}

impl Side<'_> {
    fn declared_shape(&self, name: &str) -> Option<Vec<usize>> {
        match self {
            Self::In(model) => model.input_shape(name),
            Self::Out(model) => model.output_shape(name),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::In(_) => "input",
            Self::Out(_) => "output",
        }
    }
}

#[cfg(test)]
impl Meta {
    /// A miniature geometry for pure tests: 4-sample hop, 8-point FFT (a
    /// 4-sample left pad), a 12-sample window in 5 rows, 4 encoder frames
    /// of 3 mel frames each, and a 3-piece vocabulary. Not a valid
    /// production geometry — `validate` is deliberately not run on it.
    pub(crate) fn tiny() -> Self {
        Self {
            sample_rate: OUTPUT_SAMPLE_RATE,
            io_precision: "float16".to_owned(),
            n_samples: 12,
            n_padded_samples: 20,
            n_fft: 8,
            hop_length: 4,
            preemph: 0.5,
            n_rows: 5,
            n_mels: 2,
            valid_frames: 3,
            enc_frames: 4,
            pred_hidden: 2,
            pred_layers: 2,
            joint_hidden: 3,
            decode_width: 8,
            vocab_size: 3,
            blank_idx: 3,
            n_logits: 9,
            num_durations: 5,
            durations: vec![0, 1, 2, 3, 4],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Production `meta.json` as shipped, so `validate` is exercised
    /// without a model on disk.
    const SHIPPED: &str = r#"{"sample_rate":16000,"window_seconds":15.0,"n_samples":240000,
        "hop_length":160,"n_rows":1503,"n_mels":128,"valid_frames":1500,"enc_frames":188,
        "pred_hidden":640,"pred_layers":2,"joint_hidden":640,"n_logits":8198,
        "vocab_size":8192,"blank_idx":8192,"num_durations":5,"durations":[0,1,2,3,4],
        "precision":"float16","io_precision":"float16","deployment_target":"iOS18",
        "encoder_chunks":2,"decode_width":8,"n_fft":512,"preemph":0.97,
        "n_padded_samples":240480}"#;

    fn shipped() -> Meta {
        serde_json::from_str(SHIPPED).expect("the shipped meta.json parses")
    }

    #[test]
    fn the_shipped_meta_is_valid_and_yields_the_lstm_state_length() {
        let meta = shipped();
        meta.validate().expect("valid");
        assert_eq!(meta.state_len(), 1280);
    }

    #[test]
    fn a_meta_whose_numbers_contradict_each_other_is_a_load_failure() {
        let mut wrong_blank = shipped();
        wrong_blank.blank_idx = 8191;
        let mut wrong_logits = shipped();
        wrong_logits.n_logits = 8197;
        let mut wrong_window = shipped();
        wrong_window.n_samples = 160_000;
        for meta in [wrong_blank, wrong_logits, wrong_window] {
            assert!(
                matches!(meta.validate(), Err(EngineError::LoadFailed(_))),
                "{meta:?} must be rejected"
            );
        }
    }
}
