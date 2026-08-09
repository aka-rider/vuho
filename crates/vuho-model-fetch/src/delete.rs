//! [`delete`] — removing a model Vuho downloaded, and nothing else
//! (ADR-020).
//!
//! The central invariant [`crate::availability`] documents cuts both ways:
//! Vuho verifies only bytes it fetched, and it deletes only bytes it
//! fetched. A model resolved from `VUHO_MODEL_FOLDER`, from the `.app`
//! bundle, or from the workspace `models/` dev tree belongs to whoever
//! provisioned it — deleting one would silently destroy an operator's
//! override, a shipped DMG's payload, or a developer's checkout.

use std::path::Path;

use vuho_model_paths::{ModelPathError, ModelSource, Resolved};

use crate::error::FetchError;
use crate::partial::{partial_dir_for, remove_partial_dir};

/// Delete `model_id`'s downloaded directory, together with any `.partial`
/// sibling left by an interrupted download.
///
/// # Errors
///
/// Returns [`FetchError::UnknownModel`] for an id absent from the embedded
/// manifest, [`FetchError::InvalidDirName`] when the directory name is not
/// a single plain path component, [`FetchError::NoUserModelsDir`] when
/// `$HOME` is unset, [`FetchError::NotDeletable`] when the model does not
/// resolve to a tree Vuho downloaded, and [`FetchError::Io`] when the
/// removal itself fails.
pub fn delete(model_id: &str) -> Result<(), FetchError> {
    let spec = vuho_model_paths::manifest()
        .stt
        .spec_for(model_id)
        .ok_or_else(|| FetchError::UnknownModel(model_id.to_owned()))?;
    let user_dir = vuho_model_paths::user_models_dir().ok_or(FetchError::NoUserModelsDir)?;
    let resolved = match vuho_model_paths::resolve_model(&spec) {
        Ok(resolved) => resolved,
        // An unusable name is refused as itself, not reported as "not
        // installed" — the operator's mistake is the name, and the whole
        // point of refusing here is that this path ends in `remove_dir_all`.
        Err(ModelPathError::InvalidDirName(invalid)) => {
            return Err(FetchError::InvalidDirName(invalid))
        }
        Err(e @ ModelPathError::NotFound { .. }) => {
            return Err(FetchError::NotDeletable(format!(
                "{model_id} is not installed: {e}"
            )))
        }
    };

    delete_resolved(model_id, &resolved, &user_dir)
}

/// The removal itself, with the resolver's answer and the user directory
/// passed in — isolated this way so tests exercise every refusal branch
/// against fabricated temporary directories, never the real
/// `~/Library/Application Support` and never the workspace `models/` tree.
fn delete_resolved(model_id: &str, resolved: &Resolved, user_dir: &Path) -> Result<(), FetchError> {
    if resolved.source != ModelSource::UserData {
        return Err(FetchError::NotDeletable(format!(
            "{model_id} was provisioned outside Vuho ({:?}) — only a model Vuho downloaded can be deleted",
            resolved.source
        )));
    }
    // The user-data candidate is built by joining onto `user_dir`, so this
    // is belt-and-braces rather than a second derivation of the same fact —
    // and it is the last line of defence before a recursive delete.
    if !resolved.path.starts_with(user_dir) {
        return Err(FetchError::NotDeletable(format!(
            "{model_id} resolved to {}, which is outside the user model directory {}",
            resolved.path.display(),
            user_dir.display()
        )));
    }

    std::fs::remove_dir_all(&resolved.path)?;
    remove_partial_dir(&partial_dir_for(&resolved.path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// A model directory plus a `.partial` sibling, both populated, under a
    /// fabricated user model directory inside a tempdir.
    fn fabricate(user_dir: &Path) -> PathBuf {
        let model_dir = user_dir.join("sample-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("weights.bin"), b"1234").unwrap();
        let partial = partial_dir_for(&model_dir);
        fs::create_dir_all(&partial).unwrap();
        fs::write(partial.join("weights.bin"), b"half").unwrap();
        model_dir
    }

    #[test]
    fn removes_the_user_data_dir_and_its_partial_sibling() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("Vuho/models");
        let model_dir = fabricate(&user_dir);
        let partial = partial_dir_for(&model_dir);

        delete_resolved(
            "sample-model",
            &Resolved {
                path: model_dir.clone(),
                source: ModelSource::UserData,
            },
            &user_dir,
        )
        .unwrap();

        assert!(!model_dir.exists());
        assert!(
            !partial.exists(),
            "an orphaned .partial would silently seed the next download"
        );
    }

    #[test]
    fn succeeds_when_there_is_no_partial_sibling() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("Vuho/models");
        let model_dir = user_dir.join("sample-model");
        fs::create_dir_all(&model_dir).unwrap();

        delete_resolved(
            "sample-model",
            &Resolved {
                path: model_dir.clone(),
                source: ModelSource::UserData,
            },
            &user_dir,
        )
        .unwrap();

        assert!(!model_dir.exists());
    }

    #[test]
    fn refuses_every_source_vuho_did_not_download() {
        for source in [
            ModelSource::Bundle,
            ModelSource::DevTree,
            ModelSource::EnvOverride,
        ] {
            let tmp = tempfile::tempdir().unwrap();
            let user_dir = tmp.path().join("Vuho/models");
            let elsewhere = tmp.path().join("provisioned-out-of-band/sample-model");
            fs::create_dir_all(&elsewhere).unwrap();
            fs::write(elsewhere.join("weights.bin"), b"1234").unwrap();

            let err = delete_resolved(
                "sample-model",
                &Resolved {
                    path: elsewhere.clone(),
                    source,
                },
                &user_dir,
            )
            .expect_err("a model Vuho did not download is not Vuho's to delete");

            assert!(matches!(err, FetchError::NotDeletable(_)), "{err:?}");
            assert!(
                elsewhere.join("weights.bin").exists(),
                "{source:?} tree must survive a refused delete untouched"
            );
        }
    }

    /// The path guard, exercised on its own: even a candidate tagged
    /// `UserData` is refused when it does not actually live under the user
    /// model directory, so a future resolver change cannot turn `delete`
    /// into a recursive removal of an arbitrary path.
    #[test]
    fn refuses_a_path_outside_the_user_model_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("Vuho/models");
        let outside = tmp.path().join("elsewhere/sample-model");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("weights.bin"), b"1234").unwrap();

        let err = delete_resolved(
            "sample-model",
            &Resolved {
                path: outside.clone(),
                source: ModelSource::UserData,
            },
            &user_dir,
        )
        .expect_err("a path outside the user model directory must never be removed");

        assert!(matches!(err, FetchError::NotDeletable(_)), "{err:?}");
        assert!(outside.join("weights.bin").exists());
    }

    #[test]
    fn an_unknown_model_id_is_rejected_before_any_filesystem_work() {
        let err = delete("no-such-model").expect_err("unknown ids have nothing to delete");
        assert!(matches!(err, FetchError::UnknownModel(_)), "{err:?}");
    }

    /// Sets an env var for the test and restores it on drop.
    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: the only test in this crate that touches this key.
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    /// End-to-end falsification of the traversal footgun: `..` components
    /// survive `Path::starts_with`, so a `VUHO_MODEL_NAME` carrying them
    /// would once have joined onto the user model directory, passed both of
    /// `delete_resolved`'s guards, and let `remove_dir_all` resolve the `..`
    /// onto an unrelated directory. The name is refused before any of that.
    #[test]
    fn a_traversing_directory_name_never_reaches_the_removal() {
        let manifest = vuho_model_paths::manifest();
        let _guard = EnvGuard::set(manifest.stt.env_name.as_str(), "../../../../tmp/victim");

        let err = delete(&manifest.stt.default_model)
            .expect_err("a traversing directory name must be refused");
        assert!(matches!(err, FetchError::InvalidDirName(_)), "{err:?}");
    }
}
