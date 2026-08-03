//! [`availability`] — the chokepoint deciding whether the model directory
//! [`vuho_model_paths::resolve_model_folder`] resolved is trustworthy
//! enough to load (ADR-020).
//!
//! **The central invariant: verification applies only to bytes Vuho itself
//! downloaded.** A first draft of this design ran the sidecar-and-lock
//! check against *every* resolved path. That is wrong, not merely
//! redundant: `scripts/bundle-macos.sh` copies the model into
//! `Contents/Resources/` with a bare `cp -R`, and `scripts/fetch-model.sh`
//! writes into the workspace `models/` directory the same way — neither
//! produces a sidecar. A uniform check would report [`vuho_domain::ModelStatus::Missing`]
//! for the DMG build, for `cargo run`, for `VUHO_MODEL_FOLDER`, and for
//! `test-stt-ffi`, and would offer to re-download ~474 MB on top of a
//! model already present. Those three trees are provisioned out-of-band
//! and trusted exactly as before this crate existed; only the tree under
//! [`vuho_model_paths::user_models_dir`] — the one location
//! [`crate::download`] ever writes into — is verified.

use std::path::PathBuf;

use vuho_domain::ModelStatus;
use vuho_model_paths::{Lock, ModelPathError};

use crate::verify::{self, VerifyDepth};

/// Whether the model is ready to load, and if not, why.
///
/// Calls [`vuho_model_paths::resolve_model_folder`] first — it stays the
/// one chokepoint for *where* the model lives. This function only adds a
/// trustworthiness judgment on top, scoped to the user-data candidate (see
/// this module's doc comment for why that scoping is load-bearing).
///
/// I/O errors while reading the user-data tree are never folded into
/// [`ModelStatus::Missing`] (CONSTITUTION rule 2 — don't fabricate a fact
/// the producer doesn't actually have): a permission-denied or otherwise
/// broken `~/Library/Application Support` reports [`ModelStatus::Failed`]
/// so it diagnoses itself, instead of looping through a download that
/// would fail the exact same way.
#[must_use]
pub fn availability() -> ModelStatus {
    let manifest = vuho_model_paths::manifest();
    let lock = vuho_model_paths::lock();
    let resolved = vuho_model_paths::resolve_model_folder(&manifest.stt.spec());
    classify(resolved, lock, vuho_model_paths::user_models_dir())
}

/// The decision logic behind [`availability`], with every external input
/// (the resolver's result, the lock, and `user_models_dir()`) passed in —
/// isolated this way so tests can exercise every branch, including "not
/// under the user directory at all", deterministically and without
/// touching the real filesystem or environment.
fn classify(
    resolved: Result<PathBuf, ModelPathError>,
    lock: &Lock,
    user_dir: Option<PathBuf>,
) -> ModelStatus {
    let total_bytes = lock.stt.total_bytes;

    let Ok(path) = resolved else {
        return ModelStatus::Missing { total_bytes };
    };

    // Env override, `.app` bundle, and workspace `models/` are provisioned
    // out-of-band and trusted exactly as ADR-008 always trusted them — no
    // sidecar, no lock check, unconditionally `Ready`. This is the direct
    // fix for the design error described in this module's doc comment.
    let Some(user_dir) = user_dir else {
        return ModelStatus::Ready;
    };
    if !path.starts_with(&user_dir) {
        return ModelStatus::Ready;
    }

    let sidecar = verify::sidecar_path(&path);
    match verify::verify_dir(&path, &sidecar, &lock.stt, VerifyDepth::Quick) {
        Ok(None) => ModelStatus::Ready,
        Ok(Some(problem)) => {
            // Logged, not discarded (CONSTITUTION rule 2 in spirit — don't
            // erase a fact this function actually has): a size/revision
            // mismatch versus a genuinely missing file are different root
            // causes, and without this line the only trace of a 474 MB
            // re-download offer is "Missing", unrecoverable from logs.
            log::warn!("vuho-model-fetch: user-data model tree failed verification: {problem}");
            ModelStatus::Missing { total_bytes }
        }
        Err(e) => ModelStatus::Failed {
            message: e.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use vuho_model_paths::{LockedFile, SttLock};

    fn sample_lock() -> Lock {
        Lock {
            schema_version: 1,
            stt: SttLock {
                dir_name: "sample-model".to_owned(),
                revision: "deadbeef".to_owned(),
                total_bytes: 474,
                files: vec![LockedFile {
                    path: "weights.bin".to_owned(),
                    size: 4,
                    sha256: "irrelevant-for-quick-depth".to_owned(),
                }],
            },
        }
    }

    fn path_error() -> ModelPathError {
        ModelPathError {
            tried: vec![PathBuf::from("/nonexistent")],
        }
    }

    #[test]
    fn resolver_miss_is_missing_with_locked_total_bytes() {
        let lock = sample_lock();
        let status = classify(
            Err(path_error()),
            &lock,
            Some(PathBuf::from("/home/user-dir")),
        );
        assert_eq!(status, ModelStatus::Missing { total_bytes: 474 });
    }

    /// The direct regression test for the design error this module's doc
    /// comment describes: a resolved path *outside* `user_models_dir()` —
    /// standing in for the bundle or dev candidate, both provisioned by a
    /// bare `cp -R` with no sidecar — must be `Ready` unconditionally, with
    /// no sidecar present at all.
    #[test]
    fn resolved_path_outside_user_dir_is_ready_without_any_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle_like_dir = tmp.path().join("Contents/Resources/sample-model");
        fs::create_dir_all(&bundle_like_dir).unwrap();
        // Deliberately no sidecar and no locked files written — a bare
        // `cp -R` never produces either.

        let lock = sample_lock();
        let user_dir = tmp.path().join("Library/Application Support/Vuho/models");
        let status = classify(Ok(bundle_like_dir), &lock, Some(user_dir));

        assert_eq!(status, ModelStatus::Ready);
    }

    #[test]
    fn resolved_path_outside_user_dir_is_ready_even_with_no_user_dir_concept() {
        let tmp = tempfile::tempdir().unwrap();
        let dev_dir = tmp.path().join("models/sample-model");
        fs::create_dir_all(&dev_dir).unwrap();

        let lock = sample_lock();
        // $HOME unset: `user_models_dir()` returns `None`.
        let status = classify(Ok(dev_dir), &lock, None);

        assert_eq!(status, ModelStatus::Ready);
    }

    #[test]
    fn resolved_path_under_user_dir_with_correct_sizes_is_ready() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("Vuho/models");
        let lock = sample_lock();
        let model_dir = user_dir.join(&lock.stt.dir_name);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("weights.bin"), b"1234").unwrap();
        fs::write(
            crate::verify::sidecar_path(&model_dir),
            crate::verify::sidecar_bytes(&lock.stt),
        )
        .unwrap();

        let status = classify(Ok(model_dir), &lock, Some(user_dir));
        assert_eq!(status, ModelStatus::Ready);
    }

    #[test]
    fn resolved_path_under_user_dir_with_truncated_file_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("Vuho/models");
        let lock = sample_lock();
        let model_dir = user_dir.join(&lock.stt.dir_name);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("weights.bin"), b"12").unwrap(); // truncated
        fs::write(
            crate::verify::sidecar_path(&model_dir),
            crate::verify::sidecar_bytes(&lock.stt),
        )
        .unwrap();

        let status = classify(Ok(model_dir), &lock, Some(user_dir));
        assert_eq!(status, ModelStatus::Missing { total_bytes: 474 });
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_user_dir_is_failed_not_missing() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("Vuho/models");
        let lock = sample_lock();
        let model_dir = user_dir.join(&lock.stt.dir_name);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("weights.bin"), b"1234").unwrap();
        fs::write(
            crate::verify::sidecar_path(&model_dir),
            crate::verify::sidecar_bytes(&lock.stt),
        )
        .unwrap();

        // Strip all permissions from the model directory itself: both the
        // sidecar and the locked files now live inside `model_dir`, so
        // stat'ing/reading any of them fails with `PermissionDenied`, not
        // `NotFound`.
        let original = fs::metadata(&model_dir).unwrap().permissions();
        fs::set_permissions(&model_dir, std::fs::Permissions::from_mode(0o000)).unwrap();

        // Running as root (some CI/container setups) bypasses permission
        // bits entirely, so the stat below would still succeed and this
        // test's premise wouldn't hold — skip cleanly rather than fail on
        // an environment property this test cannot control.
        let still_readable = fs::metadata(model_dir.join("weights.bin")).is_ok();
        if still_readable {
            fs::set_permissions(&model_dir, original).unwrap();
            eprintln!("skipping unreadable_user_dir_is_failed_not_missing: running as root, permission bits are not enforced");
            return;
        }

        let status = classify(Ok(model_dir.clone()), &lock, Some(user_dir));

        // Restore permissions so the tempdir can be cleaned up.
        fs::set_permissions(&model_dir, original).unwrap();

        assert!(
            matches!(status, ModelStatus::Failed { .. }),
            "expected Failed, got {status:?}"
        );
    }
}
