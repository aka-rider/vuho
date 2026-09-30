//! Parakeet-TDT: the greedy TDT decode loop, which every platform shares,
//! and its `CoreML` inference, which only macOS has.

#[cfg(target_os = "macos")]
pub(crate) mod decoder_state;
#[cfg(target_os = "macos")]
pub(crate) mod models;
pub(crate) mod tdt;
