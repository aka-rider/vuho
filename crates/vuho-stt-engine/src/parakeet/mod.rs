//! Parakeet-TDT: the greedy TDT decode loop, which every platform shares,
//! and its inference, which is `CoreML` on macOS and ONNX Runtime on Linux.

#[cfg(target_os = "macos")]
pub(crate) mod decoder_state;
#[cfg(target_os = "macos")]
pub(crate) mod models;
#[cfg(target_os = "linux")]
pub(crate) mod onnx_models;
pub(crate) mod tdt;
