//! [`download`] — fetches one manifest model from the Hub and leaves a
//! complete, fully-verified tree under [`vuho_model_paths::user_models_dir`]
//! (ADR-020).
//!
//! `hf-hub` 1.0's `local_dir` mode has no `.incomplete` staging of its own:
//! bytes stream directly to their final path inside `local_dir`, and a
//! pre-existing destination file is skipped on nothing stronger than
//! `Path::exists` (no size or hash check). If this crate reused a leftover
//! `<dir>.partial` from a previous failed attempt, a file truncated
//! mid-transfer would be silently treated as already-downloaded, and
//! [`verify_full`] would then fail the *same* way on every retry — the only
//! escape being to delete `.partial` by hand. So [`download`] removes
//! `partial_dir` before every `fetch_into` call: "the partial tree is
//! exactly what this run downloaded" is a structural invariant, not a
//! claim resumption would have to uphold on hf-hub's behalf.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use crossbeam_channel::Sender;
use vuho_domain::ModelStatus;
use vuho_model_paths::{SttLock, SttModel};

use crate::error::FetchError;
use crate::partial::{partial_dir_for, remove_partial_dir};
use crate::progress::ChannelProgress;
use crate::verify::{self, VerifyDepth};

/// Concurrent per-file transfer count handed to `hf-hub`'s
/// `snapshot_download`. `hf-hub`'s own documented default is 8; named here
/// rather than left as a bare literal (CONSTITUTION rule 27) since this
/// crate chooses it deliberately, not by omission.
const MAX_DOWNLOAD_WORKERS: usize = 8;

/// Download `model_id` into [`vuho_model_paths::user_models_dir`], verify
/// it against the embedded lock, and return the final model directory only
/// once that verification has passed.
///
/// Order, exactly:
///
/// 1. Remove any leftover `<dir>.partial` from a previous attempt (see this
///    module's doc comment), then `snapshot_download` every locked
///    component into a **fresh** `<dir>.partial`, reporting progress on
///    `progress_tx` via [`ChannelProgress`].
/// 2. Fully (hashing) verify `<dir>.partial` against the lock — reports
///    [`ModelStatus::Verifying`] on `progress_tx` first, so the UI doesn't
///    look stuck between "100% downloaded" and the final result.
/// 3. Only once that verification has passed, write the sidecar manifest
///    **inside** `<dir>.partial` via [`vuho_model_paths::atomic_write`].
///    Writing it here — after the hash check, inside the tree that is
///    about to be renamed as a unit — is what makes "sidecar present ⇒
///    these bytes were verified" a structural property: the sidecar can
///    never reach the final directory without the verified bytes it
///    describes, because the same [`fs::rename`] moves both together.
/// 4. `fs::rename` `<dir>.partial` to the final directory. The final
///    directory therefore only ever exists in a complete, verified state,
///    sidecar included — a crash at any earlier step leaves `.partial`
///    with no sidecar promoted, which [`crate::availability`] reports as
///    [`ModelStatus::Missing`], and the next call to this function removes
///    that leftover `.partial` and starts a fresh download into it.
///
/// Blocking: this function performs no async work of its own. `hf-hub`
/// owns and drives its own tokio runtime internally; this function is
/// meant to be called from a plain `std::thread`, forwarding progress to
/// the caller over `progress_tx`.
///
/// # Errors
///
/// Returns [`FetchError`] on an unknown model id, a directory name that is
/// not a single plain path component, a missing `$HOME`, a network/transfer
/// failure, an I/O failure, or a post-download verification mismatch.
pub fn download(model_id: &str, progress_tx: &Sender<ModelStatus>) -> Result<PathBuf, FetchError> {
    let manifest = vuho_model_paths::manifest();
    let (Some(model), Some(spec), Some(stt)) = (
        manifest.stt.model(model_id),
        manifest.stt.spec_for(model_id),
        vuho_model_paths::lock().model(model_id),
    ) else {
        return Err(FetchError::UnknownModel(model_id.to_owned()));
    };

    // The resolved directory name honors `VUHO_MODEL_NAME` exactly like
    // `resolve_model_folder` does — using `stt.dir_name` directly here
    // would silently ignore the override, writing bytes `availability()`
    // (which resolves the same override-aware path) would never find — and
    // it validates the name, so the `join` below cannot leave `user_dir`.
    // Ahead of `create_dir_all`, so a refused name creates nothing.
    let dir_name = vuho_model_paths::dir_name_for(&spec)?;

    let user_dir = vuho_model_paths::user_models_dir().ok_or(FetchError::NoUserModelsDir)?;
    fs::create_dir_all(&user_dir)?;

    let final_dir = user_dir.join(&dir_name);
    let partial_dir = partial_dir_for(&final_dir);
    remove_partial_dir(&partial_dir)?;
    fetch_into(&partial_dir, model, stt, progress_tx)?;

    let _ = progress_tx.send(ModelStatus::Verifying);
    finish_download(&partial_dir, &final_dir, stt)
}

/// Verify an already-fetched `partial_dir` against `stt`, write its sidecar,
/// and promote it to `final_dir` — the tail of [`download`] that runs after
/// the network transfer, factored out so it exercises entirely on the
/// filesystem and is testable without network I/O (see the tests below).
fn finish_download(
    partial_dir: &std::path::Path,
    final_dir: &std::path::Path,
    stt: &SttLock,
) -> Result<PathBuf, FetchError> {
    verify_full(partial_dir, stt)?;
    write_sidecar(partial_dir, stt)?;
    finalize(partial_dir, final_dir)?;
    Ok(final_dir.to_path_buf())
}

/// Write the sidecar manifest for `stt` inside `partial_dir`, atomically —
/// see [`download`]'s doc comment for why this must happen after
/// verification and before the `.partial` → final rename.
fn write_sidecar(partial_dir: &std::path::Path, stt: &SttLock) -> Result<(), FetchError> {
    let sidecar_path = verify::sidecar_path(partial_dir);
    vuho_model_paths::atomic_write(&sidecar_path, &verify::sidecar_bytes(stt))?;
    Ok(())
}

/// Every allow-pattern needed to select `stt`'s components out of the
/// upstream repo, derived entirely from the manifest — never a hardcoded
/// model name or file list. Each component contributes two patterns: its
/// exact repository path (matches a single file) and `"<component>/**"`
/// (matches every file inside it, for a `.mlmodelc` directory component).
/// Emitting both for every component avoids having to sniff which
/// components are files versus directories.
fn allow_patterns(components: &[&str]) -> Vec<String> {
    components
        .iter()
        .flat_map(|component| [(*component).to_owned(), format!("{component}/**")])
        .collect()
}

/// Run `hf-hub`'s `snapshot_download` for `model`'s assets into
/// `partial_dir`, reporting progress on `progress_tx`.
fn fetch_into(
    partial_dir: &std::path::Path,
    model: &SttModel,
    stt: &SttLock,
    progress_tx: &Sender<ModelStatus>,
) -> Result<(), FetchError> {
    let (owner, name) = hf_hub::split_id(&model.repo);
    let patterns = allow_patterns(&model.components());
    let handler = Arc::new(ChannelProgress::new(progress_tx.clone(), stt.total_bytes));

    let client = hf_hub::HFClientSync::new()?;
    let repo = client.model(owner, name);
    repo.snapshot_download()
        .revision(stt.revision.clone())
        .allow_patterns(patterns)
        .local_dir(partial_dir.to_path_buf())
        .max_workers(MAX_DOWNLOAD_WORKERS)
        .progress(handler)
        .send()?;
    Ok(())
}

/// Full (hashing) verification of `partial_dir` against `stt`, immediately
/// after a download completes.
fn verify_full(partial_dir: &std::path::Path, stt: &SttLock) -> Result<(), FetchError> {
    match verify::verify_files(partial_dir, stt, VerifyDepth::Full)? {
        None => Ok(()),
        Some(problem) => Err(FetchError::Verification(problem.to_string())),
    }
}

/// Replace `final_dir` with the now-verified `partial_dir`.
///
/// `fs::rename` fails on most platforms when the destination is a
/// non-empty directory, so a stale `final_dir` (e.g. left over from a
/// tampered or manually-modified state) is removed first. This function
/// only ever runs after [`verify_full`] has passed, so `final_dir` is
/// replaced only by a tree already proven complete and correct.
fn finalize(partial_dir: &std::path::Path, final_dir: &std::path::Path) -> Result<(), FetchError> {
    if final_dir.exists() {
        fs::remove_dir_all(final_dir)?;
    }
    fs::rename(partial_dir, final_dir)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use vuho_model_paths::LockedFile;

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex::encode(hasher.finalize())
    }

    /// A single-file lock naming `weights.bin` with `content`'s real size
    /// and hash, tagged with `revision`.
    fn lock_for(revision: &str, content: &[u8]) -> SttLock {
        SttLock {
            dir_name: "sample-model".to_owned(),
            revision: revision.to_owned(),
            total_bytes: content.len() as u64,
            files: vec![LockedFile {
                path: "weights.bin".to_owned(),
                size: content.len() as u64,
                sha256: sha256_hex(content),
            }],
        }
    }

    #[test]
    fn allow_patterns_covers_files_and_directories_for_every_component() {
        let components = ["Preprocessor.mlmodelc", "vocab.json"];
        let patterns = allow_patterns(&components);
        assert!(patterns.contains(&"Preprocessor.mlmodelc".to_owned()));
        assert!(patterns.contains(&"Preprocessor.mlmodelc/**".to_owned()));
        assert!(patterns.contains(&"vocab.json".to_owned()));
        assert!(patterns.contains(&"vocab.json/**".to_owned()));
        assert_eq!(patterns.len(), components.len() * 2);
    }

    #[test]
    fn finalize_replaces_a_stale_final_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let partial = tmp.path().join("model.partial");
        let final_dir = tmp.path().join("model");
        fs::create_dir_all(&partial).unwrap();
        fs::write(partial.join("new.bin"), b"new").unwrap();
        fs::create_dir_all(&final_dir).unwrap();
        fs::write(final_dir.join("stale.bin"), b"stale").unwrap();

        finalize(&partial, &final_dir).unwrap();

        assert!(!partial.exists());
        assert!(final_dir.join("new.bin").exists());
        assert!(!final_dir.join("stale.bin").exists());
    }

    // ── Blocker 2: the sidecar must only ever describe verified bytes ──

    /// Falsification target for Blocker 2. A failed verification must
    /// leave no sidecar behind anywhere a caller could find it — neither a
    /// freshly-created one inside the still-present `.partial` nor, most
    /// importantly, a corrupted one that overwrites a previously-verified
    /// revision's sidecar in `final_dir`.
    ///
    /// The pre-fix `download()` wrote the sidecar unconditionally at the
    /// very start of the function, before a single byte was fetched — so a
    /// failed verification (this test's scenario) would already have
    /// clobbered `final_dir`'s sidecar with the *new* revision's metadata
    /// while `final_dir` still held the *old* revision's bytes. `git
    /// stash`ing the fix removes `finish_download`, so this test fails to
    /// compile against the pre-fix code — the direct consequence of the
    /// sidecar write no longer being a step this function's tail owns at
    /// all.
    #[test]
    fn failed_verification_writes_no_sidecar_and_does_not_touch_final_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let final_dir = tmp.path().join("sample-model");
        let partial_dir = partial_dir_for(&final_dir);

        // An existing, correctly-verified older revision already sits at
        // `final_dir`.
        let lock_a = lock_for("revision-a", b"AAAAA");
        fs::create_dir_all(&final_dir).unwrap();
        fs::write(final_dir.join("weights.bin"), b"AAAAA").unwrap();
        fs::write(
            verify::sidecar_path(&final_dir),
            verify::sidecar_bytes(&lock_a),
        )
        .unwrap();

        // A new download attempt for revision B drops mid-transfer: the
        // `.partial` tree does not match what the lock for B expects.
        let lock_b = lock_for("revision-b", b"BBBBB");
        fs::create_dir_all(&partial_dir).unwrap();
        fs::write(partial_dir.join("weights.bin"), b"WRONG-BYTES").unwrap();

        let err = finish_download(&partial_dir, &final_dir, &lock_b)
            .expect_err("mismatched bytes must fail verification");
        assert!(matches!(err, FetchError::Verification(_)));

        // No sidecar was ever written into the still-present `.partial`.
        assert!(
            !verify::sidecar_path(&partial_dir).exists(),
            "a failed verification must never produce a sidecar"
        );

        // And critically: the pre-existing, verified revision A tree in
        // `final_dir` is completely untouched — same bytes, same sidecar
        // still naming revision A, not revision B.
        assert_eq!(
            fs::read(final_dir.join("weights.bin")).unwrap(),
            b"AAAAA",
            "final_dir's bytes must survive an unrelated failed download attempt"
        );
        let sidecar_bytes = fs::read(verify::sidecar_path(&final_dir)).unwrap();
        let sidecar: SttLock = serde_json::from_slice(&sidecar_bytes).unwrap();
        assert_eq!(
            sidecar.revision, "revision-a",
            "final_dir's sidecar must still name the revision actually on disk"
        );
    }

    /// The positive case: once `partial_dir` genuinely matches the lock,
    /// `finish_download` writes the sidecar and promotes it, and the
    /// sidecar it writes lives inside the final directory (so it moved
    /// with the same rename as the bytes it describes).
    #[test]
    fn successful_verification_writes_the_sidecar_and_finalizes() {
        let tmp = tempfile::tempdir().unwrap();
        let final_dir = tmp.path().join("sample-model");
        let partial_dir = partial_dir_for(&final_dir);
        let lock = lock_for("revision-a", b"AAAAA");
        fs::create_dir_all(&partial_dir).unwrap();
        fs::write(partial_dir.join("weights.bin"), b"AAAAA").unwrap();

        let result = finish_download(&partial_dir, &final_dir, &lock).unwrap();

        assert_eq!(result, final_dir);
        assert!(!partial_dir.exists());
        let sidecar_bytes = fs::read(verify::sidecar_path(&final_dir)).unwrap();
        let sidecar: SttLock = serde_json::from_slice(&sidecar_bytes).unwrap();
        assert_eq!(sidecar.revision, "revision-a");
    }
}
