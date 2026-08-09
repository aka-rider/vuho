//! A real, network-touching download of the Parakeet-TDT model — **not**
//! run by default (`cargo test`), only via `cargo test -p vuho-model-fetch
//! --test real_download -- --ignored`, matching how `vuho-stt-engine`'s
//! `streaming_smoke` is handled.
//!
//! Downloads into a scratch `$HOME` (a tempdir, never the real
//! `~/Library/Application Support` and never the workspace `models/`
//! directory) by overriding the `HOME` env var for the duration of this
//! process — [`vuho_model_paths::user_models_dir`] reads `$HOME` directly
//! and is not otherwise injectable, and this crate does not own that
//! function to change that. Safe here because this test is `#[ignore]`d
//! and meant to be run alone, not interleaved with other tests that read
//! `$HOME`.

use std::fs;

use sha2::{Digest, Sha256};

#[test]
#[ignore = "touches the network and downloads ~474 MB; run manually"]
fn downloads_and_fully_verifies_the_real_model() {
    let scratch_home = tempfile::tempdir().expect("create scratch HOME");
    // SAFETY: this test is `#[ignore]`d and documented to run alone
    // (`--ignored`, not interleaved with the default suite), so no other
    // thread in this process observes `$HOME` concurrently.
    unsafe {
        std::env::set_var("HOME", scratch_home.path());
    }

    let model_id = &vuho_model_paths::manifest().stt.default_model;
    let (tx, rx) = crossbeam_channel::unbounded();
    let started = std::time::Instant::now();
    let result = vuho_model_fetch::download(model_id, &tx);
    let elapsed = started.elapsed();

    let progress_events: Vec<_> = rx.try_iter().collect();
    println!(
        "real_download: {} progress events, wall-clock {:.1}s",
        progress_events.len(),
        elapsed.as_secs_f64()
    );

    let final_dir = result.expect("download should succeed against the real Hub");

    let lock = vuho_model_paths::lock()
        .model(model_id)
        .unwrap_or_else(|| panic!("{model_id} must be locked"));
    assert_eq!(
        lock.files.len(),
        20,
        "lock should still name exactly 20 files"
    );

    for locked in &lock.files {
        let path = final_dir.join(&locked.path);
        let bytes = fs::read(&path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
        assert_eq!(
            bytes.len() as u64,
            locked.size,
            "{} size mismatch",
            locked.path
        );

        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = hex::encode(hasher.finalize());
        assert_eq!(hash, locked.sha256, "{} sha256 mismatch", locked.path);
    }

    println!(
        "real_download: all {} locked files present at the correct size and hash under {:?}",
        lock.files.len(),
        final_dir
    );
}
