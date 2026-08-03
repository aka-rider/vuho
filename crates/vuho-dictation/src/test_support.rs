//! Shared test-only helpers for `vuho-dictation`'s unit tests.
//!
//! Kept in one place (rule 26) rather than duplicated across `pipeline.rs`'s
//! `#[cfg(test)] mod tests` — tests need a fake `Injector` that never calls
//! the real CGEvent/clipboard APIs (calling those from a non-main test thread
//! crashes with SIGTRAP rather than failing cleanly).

use std::sync::{Arc, Mutex};

use crate::Injector;

/// A fake [`Injector`] for tests: records every injected string (so a test
/// can assert on what was "sent") and always reports success. Never touches
/// the clipboard or `CGEvent` APIs.
pub(crate) fn fake_injector() -> (Injector, Arc<Mutex<Vec<String>>>) {
    let received: Arc<Mutex<Vec<String>>> = Arc::default();
    let received_for_closure = Arc::clone(&received);
    let injector: Injector = Arc::new(move |text: &str| {
        received_for_closure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(text.to_owned());
        Ok(())
    });
    (injector, received)
}
