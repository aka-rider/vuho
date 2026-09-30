//! The host-side half of voz's front end: everything the mel and encoder
//! bundles expect the caller to have prepared. Pure, so the layout — the
//! part a `CoreML` shape check cannot catch — is unit-tested without a
//! model.

use super::meta::{Meta, ENCODER_SUBSAMPLING};

/// Attention bias for an encoder frame that carries no real audio.
const PAD_BIAS: f32 = -40_000.0;

/// The window as the mel bundle's `audio_rows` input, row-major
/// `[1, hop_length, 1, n_rows]`.
///
/// The bundle does no framing of its own: the host lays the padded signal
/// out so that sample `i` lands at row `i % hop_length`, column
/// `i / hop_length`.
pub(crate) fn audio_rows(samples: &[f32], meta: &Meta) -> Vec<f32> {
    let signal = padded_signal(samples, meta);
    let mut rows = vec![0.0; signal.len()];
    for (i, &v) in signal.iter().enumerate() {
        rows[(i % meta.hop_length) * meta.n_rows + i / meta.hop_length] = v;
    }
    rows
}

/// `samples` as the STFT sees them: `n_fft / 2` zeros on the left (the
/// reference STFT centers its frames), then the audio, then the tail.
///
/// The tail is the pre-emphasis filter's response to the last sample — a
/// geometric decay — but only for a window that is exactly full. A partial
/// window (a streaming partial) ends in silence, so its tail is zero.
fn padded_signal(samples: &[f32], meta: &Meta) -> Vec<f32> {
    let left_pad = meta.n_fft / 2;
    let mut signal = vec![0.0; meta.n_padded_samples];
    let audio_end = left_pad + samples.len();
    signal[left_pad..audio_end].copy_from_slice(samples);

    let last = samples.last().copied().unwrap_or(0.0);
    if samples.len() == meta.n_samples && last != 0.0 {
        let mut decay = 1.0;
        for slot in &mut signal[audio_end..] {
            decay *= meta.preemph;
            *slot = decay * last;
        }
    }
    signal
}

/// Mel frames that carry real audio for `sample_count` samples: at least
/// one, at most `valid_frames`.
fn mel_valid_frames(sample_count: usize, meta: &Meta) -> usize {
    (sample_count / meta.hop_length + 1).clamp(1, meta.valid_frames)
}

/// The mel bundle's `mel_mask` input, `[1, 1, 1, valid_frames]`: 1 for a
/// frame with real audio, 0 after.
pub(crate) fn mel_mask(sample_count: usize, meta: &Meta) -> Vec<f32> {
    let valid = mel_valid_frames(sample_count, meta);
    (0..meta.valid_frames)
        .map(|frame| f32::from(u8::from(frame < valid)))
        .collect()
}

/// The encoder's `key_bias` input, `[1, enc_frames, 1, 1]`: 0 for an
/// encoder frame that attends to real audio, a large negative number for
/// one that would only attend to padding.
pub(crate) fn key_bias(sample_count: usize, meta: &Meta) -> Vec<f32> {
    let samples_per_frame = meta.hop_length * ENCODER_SUBSAMPLING;
    let attended = sample_count
        .div_ceil(samples_per_frame)
        .clamp(1, meta.enc_frames);
    (0..meta.enc_frames)
        .map(|frame| if frame < attended { 0.0 } else { PAD_BIAS })
        .collect()
}

/// The encoder's `pad_mask` input, `[1, 1, 1, enc_frames]`.
///
/// All ones **whatever the audio length** — vendor quirk: zeroing the
/// padded positions makes the encoder's `BatchNorm` blow up. Padding is
/// masked by [`key_bias`] instead.
pub(crate) fn pad_mask(meta: &Meta) -> Vec<f32> {
    vec![1.0; meta.enc_frames]
}

/// How many encoder frames the decode may visit: those that cover real
/// mel frames, never more than the encoder emits.
pub(crate) fn valid_encoder_frames(sample_count: usize, meta: &Meta) -> usize {
    mel_valid_frames(sample_count, meta)
        .div_ceil(ENCODER_SUBSAMPLING)
        .min(meta.enc_frames)
}

/// The decoder's `enc_step` input, row-major `[1, joint_hidden, 1,
/// decode_width]`: `span` consecutive encoder frames starting at
/// `position`, zero-padded to `decode_width`.
///
/// `projected` is the encoder's output frame-major: frame `t` is
/// `projected[t * joint_hidden..][..joint_hidden]`.
pub(crate) fn enc_step(projected: &[f32], position: usize, span: usize, meta: &Meta) -> Vec<f32> {
    let channels = meta.joint_hidden;
    let width = meta.decode_width;
    let mut step = vec![0.0; channels * width];
    for lane in 0..span {
        let frame = &projected[(position + lane) * channels..][..channels];
        for (channel, &v) in frame.iter().enumerate() {
            step[channel * width + lane] = v;
        }
    }
    step
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows_of(samples: &[f32]) -> Vec<Vec<f32>> {
        let meta = Meta::tiny();
        audio_rows(samples, &meta)
            .chunks(meta.n_rows)
            .map(<[f32]>::to_vec)
            .collect()
    }

    /// Sample `i` of the padded signal lands at row `i % hop`, column
    /// `i / hop`, after `n_fft / 2` zeros of left pad.
    #[test]
    fn audio_rows_lays_the_left_padded_signal_out_by_hop() {
        let rows = rows_of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(rows.len(), 4, "one row per hop sample");
        assert_eq!(rows[0], [0.0, 1.0, 5.0, 0.0, 0.0]);
        assert_eq!(rows[1], [0.0, 2.0, 6.0, 0.0, 0.0]);
        assert_eq!(rows[2], [0.0, 3.0, 0.0, 0.0, 0.0]);
        assert_eq!(rows[3], [0.0, 4.0, 0.0, 0.0, 0.0]);
    }

    /// A full window continues past its last sample with the pre-emphasis
    /// decay: `preemph^k * last` for `k = 1, 2, ...`.
    #[test]
    fn a_full_window_ends_in_the_pre_emphasis_decay_of_its_last_sample() {
        let meta = Meta::tiny();
        let samples: Vec<f32> = (1u8..=12).map(f32::from).collect();
        let signal = padded_signal(&samples, &meta);
        assert_eq!(signal.len(), 20);
        assert_eq!(&signal[16..], [6.0, 3.0, 1.5, 0.75]);
    }

    #[test]
    fn a_full_window_that_ends_in_silence_has_no_tail() {
        let meta = Meta::tiny();
        let mut samples = vec![1.0; 12];
        samples[11] = 0.0;
        assert_eq!(&padded_signal(&samples, &meta)[16..], [0.0; 4]);
    }

    #[test]
    fn a_partial_window_has_no_tail() {
        let meta = Meta::tiny();
        let signal = padded_signal(&[1.0; 11], &meta);
        assert_eq!(&signal[15..], [0.0; 5]);
    }

    /// Mel frames: `n / hop + 1`, at least 1, at most `valid_frames`.
    #[test]
    fn the_mel_mask_counts_real_frames_for_a_short_and_a_full_window() {
        let meta = Meta::tiny();
        assert_eq!(
            mel_mask(0, &meta),
            [1.0, 0.0, 0.0],
            "silence keeps one frame"
        );
        assert_eq!(mel_mask(6, &meta), [1.0, 1.0, 0.0], "6 / 4 + 1 = 2");
        assert_eq!(
            mel_mask(12, &meta),
            [1.0, 1.0, 1.0],
            "capped at valid_frames"
        );
    }

    /// Encoder frames attended: `ceil(n / (hop * 8))`, at least 1.
    #[test]
    fn the_key_bias_attends_the_frames_that_hold_audio() {
        let meta = Meta::tiny();
        let attended = |n| {
            key_bias(n, &meta)
                .iter()
                .filter(|&&bias| bias == 0.0)
                .count()
        };
        assert_eq!(attended(0), 1, "at least one");
        assert_eq!(attended(32), 1, "exactly one encoder frame of audio");
        assert_eq!(attended(33), 2, "one sample into the next");
        assert_eq!(attended(12), 1);
        let biased = key_bias(0, &meta);
        assert_eq!(biased, [0.0, PAD_BIAS, PAD_BIAS, PAD_BIAS]);
    }

    #[test]
    fn the_pad_mask_is_all_ones() {
        assert_eq!(pad_mask(&Meta::tiny()), [1.0; 4]);
    }

    /// `valid_encoder_frames = ceil(mel_valid / 8)`, capped at `enc_frames`.
    #[test]
    fn the_decode_limit_is_the_encoder_frames_over_real_mel_frames() {
        let meta = Meta::tiny();
        assert_eq!(valid_encoder_frames(0, &meta), 1);
        assert_eq!(valid_encoder_frames(12, &meta), 1, "3 mel frames");
        let mut long = Meta::tiny();
        long.valid_frames = 20;
        long.enc_frames = 2;
        assert_eq!(valid_encoder_frames(76, &long), 2, "20 mel frames, capped");
        assert_eq!(valid_encoder_frames(28, &long), 1, "8 mel frames");
        assert_eq!(valid_encoder_frames(32, &long), 2, "9 mel frames");
    }

    /// Frame-major projection in, channel-major padded step out.
    #[test]
    fn enc_step_transposes_a_span_and_zero_pads_the_rest() {
        let meta = Meta::tiny();
        let projected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let step = enc_step(&projected, 1, 2, &meta);
        assert_eq!(step.len(), 3 * 8);
        assert_eq!(step[..8], [4.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(step[8..16], [5.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(step[16..], [6.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
