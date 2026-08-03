//! [`FetchError`] — everything that can prevent [`crate::download`] from
//! producing a complete, verified model directory.

/// The error space for [`crate::download`]: network / verification / I/O,
/// kept as three distinct variant families so a caller (and a human
/// reading a log line) can tell "retry later" (network) from "the disk is
/// broken" (I/O) from "the bytes we got don't match the lock"
/// (verification) apart at a glance.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    /// `$HOME` is unset or empty, so [`vuho_model_paths::user_models_dir`]
    /// has nowhere to place a download. Distinct from [`Self::Io`]: no
    /// filesystem call was ever attempted.
    #[error("cannot determine the user model directory ($HOME is unset)")]
    NoUserModelsDir,

    /// `hf-hub` failed to reach the Hub, resolve the pinned revision, or
    /// transfer a file (network failure, HTTP error, Xet/HTTPS transport
    /// failure).
    #[error("model download failed: {0}")]
    Network(#[from] hf_hub::HFError),

    /// A filesystem operation on the sidecar, the `.partial` directory, or
    /// the final rename failed. Not the sidecar/lock **content** being
    /// wrong — see [`Self::Verification`] for that.
    #[error("model directory I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The downloaded `.partial` tree did not match `models.lock.json`
    /// after a full (hashing) verification pass: a missing file, a size
    /// mismatch, or a SHA-256 mismatch. The message names the offending
    /// file, and for a hash mismatch, both the expected and the actual
    /// hash.
    #[error("model verification failed: {0}")]
    Verification(String),
}
