//! [`availability`] — the chokepoint deciding whether the model directory
//! [`vuho_model_paths::resolve_model`] resolved is trustworthy enough to
//! load (ADR-020).
//!
//! **The central invariant: verification applies only to bytes Vuho itself
//! downloaded.** A first draft of this design ran the sidecar-and-lock
//! check against *every* resolved path. That is wrong, not merely
//! redundant: `scripts/bundle-macos.sh` copies the model into
//! `Contents/Resources/` with a bare `cp -R`, and `scripts/fetch-model.sh`
//! writes into the workspace `models/` directory the same way — neither
//! produces a sidecar. A uniform check would report [`vuho_domain::ModelStatus::Missing`]
//! for the DMG build, for `cargo run`, for `VUHO_MODEL_FOLDER`, and for
//! `test-stt-ffi`, and would offer to re-download hundreds of megabytes on
//! top of a model already present. Those three trees are provisioned
//! out-of-band and trusted exactly as before this crate existed; only the
//! tree under [`vuho_model_paths::user_models_dir`] — the one location
//! [`crate::download`] ever writes into — is verified.
//!
//! The same invariant governs [`crate::delete`]: [`ModelAvailability::deletable`]
//! is true only for [`ModelSource::UserData`], because those are the only
//! bytes Vuho put there.

use std::path::PathBuf;

use vuho_domain::ModelStatus;
use vuho_model_paths::{ModelPathError, ModelSource, Resolved, SttLock};

use crate::os_support::min_macos_satisfied;
use crate::verify::{self, VerifyDepth};

/// One model's readiness, as the Settings UI needs to render its row.
///
/// Model identity travels here rather than inside [`ModelStatus`], which is
/// deliberately not `#[non_exhaustive]` (ADR-018) so that adding a variant
/// breaks every `match` in the workspace — a wrapper carries the id without
/// spending that budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelAvailability {
    /// Manifest model id, e.g. the key under `stt.models`.
    pub id: String,
    /// Human-readable name for the Settings UI.
    pub display_name: String,
    /// Whether the model is ready to load, and if not, why.
    pub status: ModelStatus,
    /// Which candidate location answered, or `None` when none did.
    pub source: Option<ModelSource>,
    /// Total download size from the lock, for progress totals and the
    /// "download this many MB" prompt.
    pub total_bytes: u64,
    /// Whether the running macOS meets the manifest's `min_macos` floor.
    pub supported_on_this_os: bool,
}

impl ModelAvailability {
    /// Whether Vuho may delete this model's directory.
    ///
    /// Only a tree [`crate::download`] itself wrote — i.e. under
    /// [`vuho_model_paths::user_models_dir`] — is Vuho's to remove. A
    /// bundled, dev-tree, or `VUHO_MODEL_FOLDER` model belongs to whoever
    /// provisioned it.
    #[must_use]
    pub fn deletable(&self) -> bool {
        self.status == ModelStatus::Ready && self.source == Some(ModelSource::UserData)
    }
}

/// [`ModelAvailability`] for `model_id`.
///
/// Calls [`vuho_model_paths::resolve_model`] first — it stays the one
/// chokepoint for *where* a model lives. This function only adds a
/// trustworthiness judgment on top, scoped to the user-data candidate (see
/// this module's doc comment for why that scoping is load-bearing).
///
/// **`VUHO_MODEL_FOLDER`/`VUHO_MODEL_NAME` answer for every model id, not
/// just the one they were set for** — they are manifest-wide (see
/// [`vuho_model_paths::SttManifest::spec_for`]). With `VUHO_MODEL_FOLDER`
/// set, [`availability_all`] therefore reports **all** models
/// `Ready`/`EnvOverride`, the Settings list offers all of them, and
/// selecting one whose backend does not match the files in that one
/// directory fails at load time. Deliberate: the override exists for
/// callers that point it at a single model tree and select that model in
/// the same breath.
///
/// I/O errors while reading the user-data tree are never folded into
/// [`ModelStatus::Missing`] (CONSTITUTION rule 2 — don't fabricate a fact
/// the producer doesn't actually have): a permission-denied or otherwise
/// broken `~/Library/Application Support` reports [`ModelStatus::Failed`]
/// so it diagnoses itself, instead of looping through a download that
/// would fail the exact same way.
#[must_use]
pub fn availability(model_id: &str) -> ModelAvailability {
    let manifest = vuho_model_paths::manifest();
    let (Some(model), Some(spec), Some(locked)) = (
        manifest.stt.model(model_id),
        manifest.stt.spec_for(model_id),
        vuho_model_paths::lock().model(model_id),
    ) else {
        return unknown_model(model_id);
    };

    let resolved = vuho_model_paths::resolve_model(&spec);
    let (status, source) = classify(resolved, locked, vuho_model_paths::user_models_dir());
    ModelAvailability {
        id: model_id.to_owned(),
        display_name: model.display_name.clone(),
        status,
        source,
        total_bytes: locked.total_bytes,
        supported_on_this_os: min_macos_satisfied(&model.min_macos),
    }
}

/// [`ModelAvailability`] for every model the manifest knows, the default
/// model first and the rest in manifest order — the order the Settings
/// list renders.
#[must_use]
pub fn availability_all() -> Vec<ModelAvailability> {
    let stt = &vuho_model_paths::manifest().stt;
    std::iter::once(stt.default_model.as_str())
        .chain(
            stt.models
                .keys()
                .map(String::as_str)
                .filter(|id| *id != stt.default_model),
        )
        .map(availability)
        .collect()
}

/// The answer for an id absent from the embedded manifest or lock: there is
/// nothing to resolve, nothing to download, and no size to quote.
fn unknown_model(model_id: &str) -> ModelAvailability {
    ModelAvailability {
        id: model_id.to_owned(),
        display_name: model_id.to_owned(),
        status: ModelStatus::Failed {
            message: format!("{model_id} is missing from the embedded manifest or lock"),
        },
        source: None,
        total_bytes: 0,
        supported_on_this_os: false,
    }
}

/// The decision logic behind [`availability`], with every external input
/// (the resolver's result, the lock, and `user_models_dir()`) passed in —
/// isolated this way so tests can exercise every branch, including "not
/// under the user directory at all", deterministically and without
/// touching the real filesystem or environment.
fn classify(
    resolved: Result<Resolved, ModelPathError>,
    locked: &SttLock,
    user_dir: Option<PathBuf>,
) -> (ModelStatus, Option<ModelSource>) {
    let total_bytes = locked.total_bytes;

    // A name that cannot be joined safely is a configuration fault, not a
    // model that merely isn't downloaded yet: reporting it `Missing` would
    // offer a Download button that fails the same way every time.
    let resolved = match resolved {
        Ok(resolved) => resolved,
        Err(ModelPathError::InvalidDirName(invalid)) => {
            return (
                ModelStatus::Failed {
                    message: invalid.to_string(),
                },
                None,
            )
        }
        Err(ModelPathError::NotFound { .. }) => {
            return (ModelStatus::Missing { total_bytes }, None)
        }
    };
    let source = Some(resolved.source);

    // Env override, `.app` bundle, and workspace `models/` are provisioned
    // out-of-band and trusted exactly as ADR-008 always trusted them — no
    // sidecar, no lock check, unconditionally `Ready`. This is the direct
    // fix for the design error described in this module's doc comment.
    if resolved.source != ModelSource::UserData {
        return (ModelStatus::Ready, source);
    }
    // The user-data candidate is only ever built from `user_models_dir()`,
    // so this re-derives nothing; it fails closed if that ever changes.
    let Some(user_dir) = user_dir else {
        return (ModelStatus::Ready, source);
    };
    if !resolved.path.starts_with(&user_dir) {
        return (ModelStatus::Ready, source);
    }

    let sidecar = verify::sidecar_path(&resolved.path);
    let status = match verify::verify_dir(&resolved.path, &sidecar, locked, VerifyDepth::Quick) {
        Ok(None) => ModelStatus::Ready,
        Ok(Some(problem)) => {
            // Logged, not discarded (CONSTITUTION rule 2 in spirit — don't
            // erase a fact this function actually has): a size/revision
            // mismatch versus a genuinely missing file are different root
            // causes, and without this line the only trace of a large
            // re-download offer is "Missing", unrecoverable from logs.
            log::warn!("vuho-model-fetch: user-data model tree failed verification: {problem}");
            ModelStatus::Missing { total_bytes }
        }
        Err(e) => ModelStatus::Failed {
            message: e.to_string(),
        },
    };
    (status, source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use vuho_model_paths::LockedFile;

    fn sample_lock() -> SttLock {
        SttLock {
            dir_name: "sample-model".to_owned(),
            revision: "deadbeef".to_owned(),
            total_bytes: 474,
            files: vec![LockedFile {
                path: "weights.bin".to_owned(),
                size: 4,
                sha256: "irrelevant-for-quick-depth".to_owned(),
            }],
        }
    }

    fn path_error() -> ModelPathError {
        ModelPathError::NotFound {
            tried: vec![PathBuf::from("/nonexistent")],
        }
    }

    fn resolved(path: PathBuf, source: ModelSource) -> Resolved {
        Resolved { path, source }
    }

    #[test]
    fn resolver_miss_is_missing_with_locked_total_bytes_and_no_source() {
        let lock = sample_lock();
        let outcome = classify(
            Err(path_error()),
            &lock,
            Some(PathBuf::from("/home/user-dir")),
        );
        assert_eq!(outcome, (ModelStatus::Missing { total_bytes: 474 }, None));
    }

    /// An unusable directory name diagnoses itself rather than offering a
    /// download that would fail identically every time.
    #[test]
    fn an_unusable_directory_name_is_failed_not_missing() {
        let lock = sample_lock();
        let (status, source) = classify(
            Err(ModelPathError::InvalidDirName(
                vuho_model_paths::InvalidDirName {
                    origin: "VUHO_MODEL_NAME".to_owned(),
                    value: "../../tmp/victim".to_owned(),
                },
            )),
            &lock,
            Some(PathBuf::from("/home/user-dir")),
        );
        assert_eq!(source, None);
        let ModelStatus::Failed { message } = status else {
            panic!("expected Failed, got {status:?}");
        };
        assert!(message.contains("../../tmp/victim"), "{message}");
        assert!(message.contains("VUHO_MODEL_NAME"), "{message}");
    }

    /// The direct regression test for the design error this module's doc
    /// comment describes: a model resolved from the `.app` bundle — which
    /// `bundle-macos.sh` provisions with a bare `cp -R` and which therefore
    /// has no sidecar — must be `Ready` unconditionally.
    #[test]
    fn bundle_source_is_ready_without_any_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle_like_dir = tmp.path().join("Contents/Resources/sample-model");
        fs::create_dir_all(&bundle_like_dir).unwrap();
        // Deliberately no sidecar and no locked files written — a bare
        // `cp -R` never produces either.

        let lock = sample_lock();
        let user_dir = tmp.path().join("Library/Application Support/Vuho/models");
        let outcome = classify(
            Ok(resolved(bundle_like_dir, ModelSource::Bundle)),
            &lock,
            Some(user_dir),
        );

        assert_eq!(
            outcome,
            (ModelStatus::Ready, Some(ModelSource::Bundle)),
            "a bundled model has no sidecar and must never be checked for one"
        );
    }

    #[test]
    fn dev_tree_source_is_ready_even_with_no_user_dir_concept() {
        let tmp = tempfile::tempdir().unwrap();
        let dev_dir = tmp.path().join("models/sample-model");
        fs::create_dir_all(&dev_dir).unwrap();

        let lock = sample_lock();
        // $HOME unset: `user_models_dir()` returns `None`.
        let outcome = classify(Ok(resolved(dev_dir, ModelSource::DevTree)), &lock, None);

        assert_eq!(outcome, (ModelStatus::Ready, Some(ModelSource::DevTree)));
    }

    #[test]
    fn env_override_source_is_ready_without_any_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let override_dir = tmp.path().join("somewhere/else");
        fs::create_dir_all(&override_dir).unwrap();

        let lock = sample_lock();
        let outcome = classify(
            Ok(resolved(override_dir, ModelSource::EnvOverride)),
            &lock,
            Some(tmp.path().join("Vuho/models")),
        );

        assert_eq!(
            outcome,
            (ModelStatus::Ready, Some(ModelSource::EnvOverride))
        );
    }

    #[test]
    fn user_data_source_with_correct_sizes_is_ready() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("Vuho/models");
        let lock = sample_lock();
        let model_dir = user_dir.join(&lock.dir_name);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("weights.bin"), b"1234").unwrap();
        fs::write(
            crate::verify::sidecar_path(&model_dir),
            crate::verify::sidecar_bytes(&lock),
        )
        .unwrap();

        let outcome = classify(
            Ok(resolved(model_dir, ModelSource::UserData)),
            &lock,
            Some(user_dir),
        );
        assert_eq!(outcome, (ModelStatus::Ready, Some(ModelSource::UserData)));
    }

    /// The `schema_version` bump that let `models.lock.json` hold several
    /// models reshaped only the repo-committed lock's *outer* object. The
    /// sidecar a download writes is the *inner* per-model object, whose
    /// shape did not change — so a model downloaded before the bump must
    /// still verify, rather than silently offering a re-download of a tree
    /// that is already complete and correct.
    #[test]
    fn a_pre_schema_bump_sidecar_still_verifies() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("Vuho/models");
        let lock = sample_lock();
        let model_dir = user_dir.join(&lock.dir_name);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("weights.bin"), b"1234").unwrap();
        // Written out literally, exactly as a pre-bump release left it on
        // disk — deliberately not via `sidecar_bytes`, which would make
        // this test agree with today's writer by construction.
        fs::write(
            crate::verify::sidecar_path(&model_dir),
            br#"{
  "dir_name": "sample-model",
  "revision": "deadbeef",
  "total_bytes": 474,
  "files": [
    { "path": "weights.bin", "size": 4, "sha256": "irrelevant-for-quick-depth" }
  ]
}"#,
        )
        .unwrap();

        let outcome = classify(
            Ok(resolved(model_dir, ModelSource::UserData)),
            &lock,
            Some(user_dir),
        );
        assert_eq!(outcome, (ModelStatus::Ready, Some(ModelSource::UserData)));
    }

    #[test]
    fn user_data_source_with_truncated_file_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("Vuho/models");
        let lock = sample_lock();
        let model_dir = user_dir.join(&lock.dir_name);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("weights.bin"), b"12").unwrap(); // truncated
        fs::write(
            crate::verify::sidecar_path(&model_dir),
            crate::verify::sidecar_bytes(&lock),
        )
        .unwrap();

        let outcome = classify(
            Ok(resolved(model_dir, ModelSource::UserData)),
            &lock,
            Some(user_dir),
        );
        assert_eq!(
            outcome,
            (
                ModelStatus::Missing { total_bytes: 474 },
                Some(ModelSource::UserData)
            )
        );
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_user_dir_is_failed_not_missing() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("Vuho/models");
        let lock = sample_lock();
        let model_dir = user_dir.join(&lock.dir_name);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("weights.bin"), b"1234").unwrap();
        fs::write(
            crate::verify::sidecar_path(&model_dir),
            crate::verify::sidecar_bytes(&lock),
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

        let (status, source) = classify(
            Ok(resolved(model_dir.clone(), ModelSource::UserData)),
            &lock,
            Some(user_dir),
        );

        // Restore permissions so the tempdir can be cleaned up.
        fs::set_permissions(&model_dir, original).unwrap();

        assert!(
            matches!(status, ModelStatus::Failed { .. }),
            "expected Failed, got {status:?}"
        );
        assert_eq!(source, Some(ModelSource::UserData));
    }

    #[test]
    fn deletable_only_for_a_ready_user_data_tree() {
        let ready_user_data = ModelAvailability {
            id: "sample".to_owned(),
            display_name: "Sample".to_owned(),
            status: ModelStatus::Ready,
            source: Some(ModelSource::UserData),
            total_bytes: 474,
            supported_on_this_os: true,
        };
        assert!(ready_user_data.deletable());

        for source in [
            ModelSource::Bundle,
            ModelSource::DevTree,
            ModelSource::EnvOverride,
        ] {
            let elsewhere = ModelAvailability {
                source: Some(source),
                ..ready_user_data.clone()
            };
            assert!(
                !elsewhere.deletable(),
                "{source:?} is provisioned out-of-band and is not Vuho's to delete"
            );
        }

        let missing = ModelAvailability {
            status: ModelStatus::Missing { total_bytes: 474 },
            ..ready_user_data.clone()
        };
        assert!(!missing.deletable());

        let unresolved = ModelAvailability {
            source: None,
            ..ready_user_data
        };
        assert!(!unresolved.deletable());
    }

    #[test]
    fn availability_all_lists_every_manifest_model_default_first() {
        let stt = &vuho_model_paths::manifest().stt;
        let listed: Vec<String> = availability_all().into_iter().map(|m| m.id).collect();

        assert_eq!(listed.first(), Some(&stt.default_model));
        assert_eq!(listed.len(), stt.models.len());
        for id in stt.models.keys() {
            assert!(listed.contains(id), "{id} missing from availability_all()");
        }
    }

    #[test]
    fn an_unknown_id_is_failed_and_not_deletable() {
        let unknown = availability("no-such-model");
        assert!(matches!(unknown.status, ModelStatus::Failed { .. }));
        assert_eq!(unknown.source, None);
        assert_eq!(unknown.total_bytes, 0);
        assert!(!unknown.deletable());
        assert!(!unknown.supported_on_this_os);
    }
}
