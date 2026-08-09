//! `vuho-model-fetch` — the **only** crate in the workspace permitted to
//! perform network I/O (ADR-020).
//!
//! Three responsibilities, each addressing one model by its manifest id:
//!
//! - [`availability`] / [`availability_all`] — the chokepoint deciding
//!   whether the model directory [`vuho_model_paths::resolve_model`]
//!   resolved is trustworthy enough to load. Verification applies *only*
//!   to the tree this crate itself downloaded, under
//!   [`vuho_model_paths::user_models_dir`] — see the `availability` module
//!   doc comment for the central invariant and why a uniform check on
//!   every resolved path would be wrong, not merely redundant.
//! - [`delete`] — removes a downloaded model, refusing every tree Vuho did
//!   not fetch itself: the same invariant, from the other side.
//! - [`download`] — fetches the model from the Hub (`hf-hub` 1.0,
//!   Xet-first with automatic HTTPS fallback) into a `.partial` directory,
//!   fully verifies it against `models.lock.json`, then atomically renames
//!   it into place. The final directory therefore only ever exists in a
//!   complete, verified state.
//!
//! Threading: [`download`] is a blocking call — `hf-hub` owns and drives
//! its own tokio runtime internally, and this crate's public API contains
//! no `async fn`. Callers run [`download`] from a plain `std::thread` and
//! receive progress over the `crossbeam_channel::Sender<vuho_domain::ModelStatus>`
//! passed in; no async runtime enters any of vuho's own crates.
//!
//! Nothing in this crate hardcodes a model name, revision, or component
//! list — every such fact is read from [`vuho_model_paths::manifest`] /
//! [`vuho_model_paths::lock`] (ADR-019).

mod availability;
mod delete;
mod download;
mod error;
mod os_support;
mod partial;
mod progress;
mod verify;

pub use availability::{availability, availability_all, ModelAvailability};
pub use delete::delete;
pub use download::download;
pub use error::FetchError;
