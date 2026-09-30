//! LSTM decoder state for the `CoreML` Parakeet-TDT backend.
//!
//! Holds the hidden/cell states and the cached decoder output that
//! [`super::tdt::TdtStep::score`] and `accept` thread through one window.

/// Hidden/cell state size: `(num_layers=2, batch=1, hidden_dim=640)`.
const H_C_SIZE: usize = 2 * 640;
/// Decoder output size: `[1, 1, 640]`.
const DEC_OUT_SIZE: usize = 640;

/// Decoder recurrent state.
#[derive(Debug)]
pub(crate) struct DecoderState {
    /// LSTM hidden state: `[2, 1, 640]`.
    pub(crate) h: Vec<f32>,
    /// LSTM cell state: `[2, 1, 640]`.
    pub(crate) c: Vec<f32>,
    /// Decoder output of the last decode step: `[1, 1, 640]`.
    pub(crate) dec_out: Vec<f32>,
}

impl DecoderState {
    /// Zeroed `h`/`c`/`dec_out`: not yet conditioned on any token, so a
    /// decode step still has to run before it is scored against.
    pub(crate) fn zeroed() -> Self {
        Self {
            h: vec![0.0; H_C_SIZE],
            c: vec![0.0; H_C_SIZE],
            dec_out: vec![0.0; DEC_OUT_SIZE],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_state_has_zeros() {
        let state = DecoderState::zeroed();
        assert!(state.h.iter().all(|&v| v == 0.0));
        assert!(state.c.iter().all(|&v| v == 0.0));
        assert!(state.dec_out.iter().all(|&v| v == 0.0));
    }
}
