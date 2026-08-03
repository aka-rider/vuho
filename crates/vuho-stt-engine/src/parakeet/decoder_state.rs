//! LSTM decoder state for Parakeet-TDT.
//!
//! Holds the hidden/cell states and cached decoder output that thread
//! across windows. Plain `Vec<f32>` / `i32` so it can be cloned for
//! partial re-inference (CONSTITUTION rule: idempotent re-run).

/// Hidden/cell state size: `(num_layers=2, batch=1, hidden_dim=640)`.
const H_C_SIZE: usize = 2 * 640;

/// Decoder recurrent state, cloned across windows for idempotent
/// re-inference of the open window (CONSTITUTION rule: a partial re-run
/// must never mutate the committed state — only a commit does).
///
/// `pub` (not `pub(crate)`): re-exported by `bench_support` for
/// `benches/hot_paths.rs` — fields stay `pub(crate)`.
#[derive(Debug, Clone)]
pub struct DecoderState {
    /// LSTM hidden state: `[2, 1, 640]`.
    pub(crate) h: Vec<f32>,
    /// LSTM cell state: `[2, 1, 640]`.
    pub(crate) c: Vec<f32>,
    /// Cached decoder output from the last decode step: `[1, 1, 640]`.
    /// `None` means "not yet primed" — `tdt_greedy` primes on first use.
    pub(crate) dec_out: Option<Vec<f32>>,
    /// Last emitted token id (doubles as SOS = blank on first step).
    pub(crate) last_token: i32,
}

impl DecoderState {
    /// Create a fresh (unprimed) decoder state: zeroed `h`/`c`, blank SOS.
    ///
    /// `pub` (not `pub(crate)`): re-exported by `bench_support` for
    /// `benches/hot_paths.rs`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            h: vec![0.0; H_C_SIZE],
            c: vec![0.0; H_C_SIZE],
            dec_out: None,
            // BLANK (8192) fits comfortably in i32.
            #[allow(clippy::cast_possible_wrap)]
            last_token: super::tdt::BLANK as i32,
        }
    }
}

impl Default for DecoderState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_has_zeros_and_blank() {
        let state = DecoderState::new();
        assert!(state.h.iter().all(|&v| v == 0.0));
        assert!(state.c.iter().all(|&v| v == 0.0));
        assert!(state.dec_out.is_none());
        assert_eq!(state.last_token, 8192);
    }

    #[test]
    fn clone_preserves_state() {
        let mut state = DecoderState::new();
        state.h[0] = 1.0;
        state.last_token = 42;
        let cloned = state.clone();
        assert!((cloned.h[0] - 1.0).abs() < f32::EPSILON);
        assert_eq!(cloned.last_token, 42);
    }
}
