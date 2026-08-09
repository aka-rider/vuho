//! The `<dir>.partial` sibling convention, shared by [`crate::download`]
//! (which stages a transfer there) and [`crate::delete`] (which must not
//! leave one orphaned behind a deleted model).

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::FetchError;

/// Suffix for the in-progress download directory, sibling to the final
/// model directory. Never renamed to the final name until every locked
/// file has been fully (hash) verified — see [`crate::download`].
const PARTIAL_SUFFIX: &str = ".partial";

/// `<final_dir>` with [`PARTIAL_SUFFIX`] appended to its last path
/// component — a sibling directory, so the eventual `rename` that promotes
/// it is a same-filesystem atomic move.
pub(crate) fn partial_dir_for(final_dir: &Path) -> PathBuf {
    let mut name = final_dir
        .file_name()
        .expect("model directory path always has a final component")
        .to_os_string();
    name.push(PARTIAL_SUFFIX);
    final_dir.with_file_name(name)
}

/// Remove `partial_dir` if it exists, reporting only real I/O failures.
///
/// A leftover `.partial` must never be reused by the next download:
/// `hf-hub`'s `local_dir` mode skips any destination file that merely
/// [`std::path::Path::exists`], with no size/hash check (see
/// [`crate::download`]'s module doc), so a truncated leftover would be
/// treated as already-downloaded and fail verification the same way on
/// every subsequent retry. Removing it first makes "the partial tree is
/// exactly what this run downloaded" structurally true rather than assumed.
///
/// [`crate::delete`] calls it for the same reason from the other side: a
/// deleted model whose `.partial` survived would leave the next download
/// resuming from bytes nobody verified.
pub(crate) fn remove_partial_dir(partial_dir: &Path) -> Result<(), FetchError> {
    match fs::remove_dir_all(partial_dir) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_dir_appends_suffix_to_final_component() {
        let final_dir = PathBuf::from("/home/user/Vuho/models/sample-model");
        assert_eq!(
            partial_dir_for(&final_dir),
            PathBuf::from("/home/user/Vuho/models/sample-model.partial")
        );
    }

    /// `hf-hub` skips any destination file that merely exists, so a
    /// `.partial` truncated by a prior interrupted download must be gone
    /// before the next fetch — otherwise every retry fails verification
    /// identically, with no recovery short of deleting `.partial` by hand.
    #[test]
    fn removes_a_leftover_truncated_partial() {
        let tmp = tempfile::tempdir().unwrap();
        let partial = tmp.path().join("model.partial");
        fs::create_dir_all(&partial).unwrap();
        fs::write(partial.join("weights.bin"), b"truncated-mid-transfer").unwrap();

        remove_partial_dir(&partial).unwrap();

        assert!(
            !partial.exists(),
            "a leftover .partial must be removed, not silently reused by the next download"
        );
    }

    #[test]
    fn is_a_noop_when_nothing_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let partial = tmp.path().join("model.partial");

        remove_partial_dir(&partial).expect("a missing .partial is not an error");

        assert!(!partial.exists());
    }
}
