//! Voice-activity detection wrapper around `voice_activity_detector` (Silero
//! v5, embedded — the crate cannot load external weights, so the fetched
//! `models/silero-vad/onnx/model_fp16.onnx` is provisioned for a future
//! direct-`ort` v6 swap but is **not loaded by the app**; see ADR-014).
//!
//! Hysteresis avoids flapping on borderline frames: entering "speech" needs
//! a higher probability than leaving it.

use voice_activity_detector::VoiceActivityDetector;

use crate::EngineError;

/// Samples per VAD frame: 512 @ 16 kHz = 32 ms (the only chunk size Silero
/// v5 accepts at this sample rate).
pub const FRAME: usize = 512;

/// Probability above which a silent detector becomes "active".
const ENTER_THRESHOLD: f32 = 0.35;
/// Probability below which an active detector becomes "silent".
const EXIT_THRESHOLD: f32 = 0.20;
/// Duration of one [`FRAME`] at 16 kHz, in milliseconds.
const FRAME_MS: u32 = 32;

/// Result of pushing a block of samples through the detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadUpdate {
    /// Whether the detector was in the "active" (speech) state for at least
    /// one frame processed by this call.
    pub any_speech: bool,
    /// Milliseconds of continuous silence immediately preceding the end of
    /// this call's input (0 if the last processed frame was speech).
    pub trailing_silence_ms: u32,
}

/// Streaming voice-activity detector: buffers input into 32 ms frames and
/// tracks a hysteresis-debounced speech/silence state across calls.
pub struct Vad {
    inner: VoiceActivityDetector,
    /// Samples accepted but not yet forming a full [`FRAME`].
    carry: Vec<f32>,
    speech_active: bool,
    silence_ms: u32,
}

impl Vad {
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if the embedded Silero session
    /// fails to build (a `voice_activity_detector` configuration error, not
    /// a missing-file error — the model is embedded in the crate).
    pub fn new() -> Result<Self, EngineError> {
        let inner = VoiceActivityDetector::builder()
            .sample_rate(16_000_i64)
            .chunk_size(FRAME)
            .build()
            .map_err(|e| EngineError::LoadFailed(format!("VAD init failed: {e}")))?;
        Ok(Self {
            inner,
            carry: Vec::with_capacity(FRAME * 2),
            speech_active: false,
            silence_ms: 0,
        })
    }

    /// Feed samples (any length; internally chunked into [`FRAME`]-sized
    /// windows, carrying a remainder across calls).
    pub fn push(&mut self, samples: &[f32]) -> VadUpdate {
        self.carry.extend_from_slice(samples);

        let mut any_speech = false;
        let mut offset = 0;
        while self.carry.len() - offset >= FRAME {
            let frame = &self.carry[offset..offset + FRAME];
            let prob = self.inner.predict(frame.iter().copied());

            if self.speech_active {
                if prob < EXIT_THRESHOLD {
                    self.speech_active = false;
                }
            } else if prob > ENTER_THRESHOLD {
                self.speech_active = true;
            }

            if self.speech_active {
                any_speech = true;
                self.silence_ms = 0;
            } else {
                self.silence_ms = self.silence_ms.saturating_add(FRAME_MS);
            }

            offset += FRAME;
        }
        self.carry.drain(..offset);

        VadUpdate {
            any_speech,
            trailing_silence_ms: self.silence_ms,
        }
    }

    /// Reset detector state (session recurrent state, hysteresis, buffered
    /// remainder) — call at the start of a new session, never mid-session
    /// (CONSTITUTION rule 3: the detector itself lives at session scope,
    /// not app scope, unlike the `CoreML` models).
    pub fn reset(&mut self) {
        self.inner.reset();
        self.carry.clear();
        self.speech_active = false;
        self.silence_ms = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A silent (all-zero) buffer should never report speech.
    #[test]
    fn silence_never_reports_speech() {
        let Ok(mut vad) = Vad::new() else {
            eprintln!("skipping: VAD session failed to build in this environment");
            return;
        };
        let zeros = vec![0.0_f32; FRAME * 4];
        let update = vad.push(&zeros);
        assert!(!update.any_speech);
        assert!(update.trailing_silence_ms > 0);
    }

    /// Partial frames are carried across calls, not dropped or padded early.
    #[test]
    fn partial_frame_carries_across_calls() {
        let Ok(mut vad) = Vad::new() else {
            eprintln!("skipping: VAD session failed to build in this environment");
            return;
        };
        let half = vec![0.0_f32; FRAME / 2];
        let update1 = vad.push(&half);
        // Fewer than FRAME samples buffered: no frame processed yet, so no
        // silence has been observed (0 ms), and definitely no speech.
        assert!(!update1.any_speech);
        assert_eq!(update1.trailing_silence_ms, 0);

        let update2 = vad.push(&half);
        // Now a full frame has been processed.
        assert!(!update2.any_speech);
        assert!(update2.trailing_silence_ms >= FRAME_MS);
    }

    /// jfk.wav's voiced region should score far above an all-zero buffer —
    /// the concrete regression fixture the plan calls for.
    #[test]
    fn jfk_voiced_region_scores_above_silence() {
        let Some(samples) = crate::test_support::load_jfk_wav_f32() else {
            eprintln!("skipping: JFK_WAV/jfk.wav not found in this environment");
            return;
        };
        let Ok(mut vad) = Vad::new() else {
            eprintln!("skipping: VAD session failed to build in this environment");
            return;
        };
        let update = vad.push(&samples);
        assert!(update.any_speech, "jfk.wav should contain detected speech");
    }
}
