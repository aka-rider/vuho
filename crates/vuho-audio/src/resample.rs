//! Mono resampling to [`crate::OUTPUT_SAMPLE_RATE`] via `rubato::FftFixedIn`.
//!
//! One source of truth for the resample algorithm (CONSTITUTION rule 26):
//! every capture stream goes through [`Resampler::new`], whatever the
//! device's native sample rate.

use rubato::Resampler as _;

use crate::{AudioError, OUTPUT_SAMPLE_RATE};

/// Frames per resample chunk. Arbitrary but fixed: `rubato::FftFixedIn`
/// needs a constant chunk size to build its FFT plan once.
const CHUNK_FRAMES: usize = 1024;

/// Resamples a mono f32 stream from an arbitrary device rate down (or up) to
/// [`OUTPUT_SAMPLE_RATE`]. A device already running at the output rate
/// short-circuits to a passthrough (no FFT plan, no latency).
pub(crate) enum Resampler {
    Passthrough,
    Fft {
        inner: Box<rubato::FftFixedIn<f32>>,
        input_buf: Vec<f32>,
        chunk_frames: usize,
    },
}

impl Resampler {
    pub(crate) fn new(input_rate: u32) -> Result<Self, AudioError> {
        if input_rate == OUTPUT_SAMPLE_RATE {
            return Ok(Self::Passthrough);
        }
        let inner = rubato::FftFixedIn::<f32>::new(
            input_rate as usize,
            OUTPUT_SAMPLE_RATE as usize,
            CHUNK_FRAMES,
            2,
            1,
        )
        .map_err(|e| AudioError::Resample(e.to_string()))?;
        Ok(Self::Fft {
            inner: Box::new(inner),
            input_buf: Vec::with_capacity(CHUNK_FRAMES * 2),
            chunk_frames: CHUNK_FRAMES,
        })
    }

    /// Feed mono samples in; returns zero or more full resampled chunks
    /// (buffers internally until a full `chunk_frames` window is available).
    pub(crate) fn process(&mut self, mono: &[f32]) -> Result<Vec<f32>, AudioError> {
        match self {
            Self::Passthrough => Ok(mono.to_vec()),
            Self::Fft {
                inner,
                input_buf,
                chunk_frames,
            } => {
                input_buf.extend_from_slice(mono);
                let mut out = Vec::new();
                while input_buf.len() >= *chunk_frames {
                    // Feed a borrowed slice straight from `input_buf` instead
                    // of draining+collecting into a fresh `Vec` per chunk;
                    // the buffer is only shrunk (no reallocation) after
                    // `rubato` is done reading it.
                    let waves_in = [&input_buf[..*chunk_frames]];
                    let waves_out = inner
                        .process(&waves_in, None)
                        .map_err(|e| AudioError::Resample(e.to_string()))?;
                    out.extend_from_slice(&waves_out[0]);
                    input_buf.drain(..*chunk_frames);
                }
                Ok(out)
            }
        }
    }

    /// Flush any buffered tail through `process_partial` (rubato's API for a
    /// final, short chunk) — call once at stream end.
    pub(crate) fn flush(&mut self) -> Result<Vec<f32>, AudioError> {
        match self {
            Self::Passthrough => Ok(Vec::new()),
            Self::Fft {
                inner, input_buf, ..
            } => {
                if input_buf.is_empty() {
                    return Ok(Vec::new());
                }
                let tail = std::mem::take(input_buf);
                let waves_in = [tail];
                let waves_out = inner
                    .process_partial(Some(&waves_in), None)
                    .map_err(|e| AudioError::Resample(e.to_string()))?;
                Ok(waves_out.into_iter().next().unwrap_or_default())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Synthesize a sine at `freq_hz`, sampled at `rate`, `n` frames.
    #[allow(clippy::cast_precision_loss)] // test fixture; frame counts are tiny
    fn sine(rate: u32, freq_hz: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (2.0 * PI * freq_hz * t).sin()
            })
            .collect()
    }

    /// Zero-crossing count is a robust dominant-frequency proxy independent
    /// of the resampler's own windowing/filter phase.
    fn zero_crossings(samples: &[f32]) -> usize {
        samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count()
    }

    /// One place the length-tolerance comparison lives (CONSTITUTION rule 26).
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // test fixture; lengths are tiny
    fn len_within_tolerance(actual: usize, expected: usize, tolerance: usize) -> bool {
        (actual as i64 - expected as i64).unsigned_abs() < tolerance as u64
    }

    #[test]
    fn passthrough_when_already_16k() {
        let mut r = Resampler::new(16_000).expect("construct");
        let input = sine(16_000, 440.0, 2048);
        let out = r.process(&input).expect("process");
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn resamples_48k_to_16k_preserves_length_ratio_and_frequency() {
        let mut r = Resampler::new(48_000).expect("construct");
        let n_in = 48_000; // 1 second @ 48kHz
        let input = sine(48_000, 440.0, n_in);

        let mut out = r.process(&input).expect("process");
        out.extend(r.flush().expect("flush"));

        let expected_len = 16_000; // 1 second @ 16kHz
        let tolerance = 1024 * 2; // within a couple of resample chunks
        assert!(
            len_within_tolerance(out.len(), expected_len, tolerance),
            "output length {} not within {tolerance} of expected {expected_len}",
            out.len(),
        );

        // Dominant frequency preserved: zero-crossing rate scales with
        // sample rate, not with frequency-in-samples, so the *count* over
        // one second of audio should match between input and output
        // (roughly 2 * freq crossings/sec for a sine).
        let in_crossings = zero_crossings(&input);
        let out_crossings = zero_crossings(&out);
        #[allow(clippy::cast_precision_loss)] // test fixture; crossing counts are tiny
        let ratio = out_crossings as f32 / in_crossings as f32;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "zero-crossing ratio {ratio} out of range"
        );
    }

    #[test]
    fn resamples_44_1k_to_16k() {
        let mut r = Resampler::new(44_100).expect("construct");
        let n_in = 44_100;
        let input = sine(44_100, 300.0, n_in);
        let mut out = r.process(&input).expect("process");
        out.extend(r.flush().expect("flush"));

        let expected_len = 16_000;
        let tolerance = 1024 * 2;
        assert!(
            len_within_tolerance(out.len(), expected_len, tolerance),
            "output length {} not within {tolerance} of expected {expected_len}",
            out.len(),
        );
    }

    #[test]
    fn flush_is_idempotent_and_completes_after_process() {
        let mut r = Resampler::new(48_000).expect("construct");
        let input = sine(48_000, 200.0, 500); // shorter than one chunk
        let out1 = r.process(&input).expect("process");
        assert!(out1.is_empty(), "partial chunk should not emit yet");
        let flushed = r.flush().expect("flush");
        assert!(
            !flushed.is_empty(),
            "flush should emit the buffered partial chunk"
        );
        let flushed_again = r.flush().expect("flush again");
        assert!(
            flushed_again.is_empty(),
            "second flush should be empty (buffer drained)"
        );
    }
}
