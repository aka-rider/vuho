//! Comparing an on-disk model tree against `models.lock.json`.
//!
//! Two independent checks live here, at two depths (see [`VerifyDepth`]):
//! the sidecar file that records *which* revision was downloaded, and the
//! locked file list itself (sizes always, SHA-256 only at [`VerifyDepth::Full`]).
//! Both are pure filesystem reads — no network I/O — so this module is
//! shared by [`crate::availability`] (read-only, `Quick`) and
//! [`crate::download`] (`Full`, immediately after a fetch).

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use vuho_model_paths::SttLock;

/// Read buffer size for streaming a file through SHA-256, chosen to keep
/// memory use flat regardless of file size without adding per-read syscall
/// overhead noticeable against `ParakeetEncoder_15s.mlmodelc`'s ~300 MB of
/// weights.
const SHA256_READ_BUF_BYTES: usize = 64 * 1024;

/// How thoroughly to compare an on-disk model tree against the lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VerifyDepth {
    /// Sidecar revision plus every locked file's exact size — no hashing.
    /// Cheap enough to sit on the startup path (`availability()`).
    Quick,
    /// Everything `Quick` checks, plus a SHA-256 comparison of every
    /// file's content. Costs 1-2 seconds over the full ~474 MB model, so
    /// it only runs immediately after a download, or behind a future
    /// explicit "Repair" action — never on startup.
    Full,
}

/// One concrete way an on-disk tree under `user_models_dir()` differs from
/// what `models.lock.json` says a complete, untampered download must
/// contain. Every variant is treated identically by both callers: as
/// "this tree cannot be trusted", never "verified missing" versus "verified
/// wrong" — the safe reaction (offer/require a fresh download) is the same
/// either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VerifyProblem {
    /// The sidecar manifest (written by [`crate::download`] only after a
    /// full verification pass, inside the model directory) does not exist.
    SidecarMissing,
    /// The sidecar exists but is not valid JSON in the expected shape.
    SidecarUnparsable,
    /// The sidecar names a different pinned revision than the embedded
    /// lock — e.g. a stale download left over from an older release.
    RevisionMismatch { expected: String, found: String },
    /// A file the lock requires is absent from the tree.
    FileMissing { path: String },
    /// A file exists but its size does not match the lock.
    SizeMismatch {
        path: String,
        expected: u64,
        found: u64,
    },
    /// A file's content hash does not match the lock (`Full` depth only).
    HashMismatch {
        path: String,
        expected: String,
        found: String,
    },
}

impl fmt::Display for VerifyProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SidecarMissing => write!(f, "sidecar manifest is missing"),
            Self::SidecarUnparsable => write!(f, "sidecar manifest is not valid JSON"),
            Self::RevisionMismatch { expected, found } => write!(
                f,
                "sidecar revision {found} does not match locked revision {expected}"
            ),
            Self::FileMissing { path } => write!(f, "{path} is missing"),
            Self::SizeMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "{path} has size {found} bytes, expected {expected} bytes"
            ),
            Self::HashMismatch {
                path,
                expected,
                found,
            } => write!(f, "{path} has sha256 {found}, expected {expected}"),
        }
    }
}

/// Filename of the sidecar manifest, inside the model directory itself.
const SIDECAR_FILENAME: &str = ".vuho-model.manifest.json";

/// The sidecar manifest path for the model directory `dir`.
///
/// Lives **inside** `dir`, not as a sibling — [`crate::download`] writes it
/// only after a full verification pass, into `<dir>.partial`, so the same
/// `fs::rename` that promotes `.partial` to the final directory carries the
/// sidecar along with it atomically. That is what makes "sidecar present
/// and parses ⇒ these exact bytes were verified" structurally true: there
/// is no filesystem state in which the sidecar exists without the verified
/// tree it describes, or describes a tree other than the one it sits inside.
pub(crate) fn sidecar_path(dir: &Path) -> PathBuf {
    dir.join(SIDECAR_FILENAME)
}

/// Serialize `stt` into the sidecar's on-disk JSON shape.
///
/// Built field-by-field with `serde_json::json!` rather than `#[derive(Serialize)]`
/// on [`SttLock`] itself: that type lives in `vuho-model-paths`, which this
/// crate does not own, and it only ever needs to be *read* back there today
/// (`serde::Deserialize`). The field names below match `SttLock`/`LockedFile`'s
/// `Deserialize` impl exactly, so [`sidecar_path`]'s file round-trips through
/// [`serde_json::from_slice::<SttLock>`] unchanged.
///
/// # Panics
///
/// Never — every value serialized here is a primitive or a `String`, which
/// `serde_json` cannot fail to encode.
pub(crate) fn sidecar_bytes(stt: &SttLock) -> Vec<u8> {
    let value = serde_json::json!({
        "dir_name": stt.dir_name,
        "revision": stt.revision,
        "total_bytes": stt.total_bytes,
        "files": stt.files.iter().map(|file| serde_json::json!({
            "path": file.path,
            "size": file.size,
            "sha256": file.sha256,
        })).collect::<Vec<_>>(),
    });
    serde_json::to_vec_pretty(&value).expect("serializing primitive JSON values cannot fail")
}

/// [`fs::read`], mapping a not-found file to `Ok(None)` instead of an
/// error — the distinction [`crate::availability`] needs between "this
/// tree is incomplete" (`Missing`) and "this tree is unreadable"
/// (`Failed`, CONSTITUTION rule 2).
fn read_optional(path: &Path) -> io::Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// [`fs::metadata`], mapping a not-found file to `Ok(None)` — see
/// [`read_optional`].
fn metadata_optional(path: &Path) -> io::Result<Option<fs::Metadata>> {
    match fs::metadata(path) {
        Ok(meta) => Ok(Some(meta)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Stream `path` through SHA-256 and return the lowercase hex digest.
fn sha256_hex_file(path: &Path) -> io::Result<String> {
    use std::io::Read;

    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; SHA256_READ_BUF_BYTES];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Compare every file the lock names against `dir`, at `depth`. Does not
/// look at the sidecar — see [`verify_dir`] for the combined check
/// [`crate::availability`] needs.
///
/// Returns `Ok(None)` when every locked file is present and matches;
/// `Ok(Some(problem))` naming the first mismatch found; `Err` only for an
/// I/O failure that is not "file not found" (permission denied, disk
/// error, …) — those are never silently folded into "missing"
/// (CONSTITUTION rule 2).
pub(crate) fn verify_files(
    dir: &Path,
    stt: &SttLock,
    depth: VerifyDepth,
) -> io::Result<Option<VerifyProblem>> {
    for file in &stt.files {
        let path = dir.join(&file.path);
        let Some(meta) = metadata_optional(&path)? else {
            return Ok(Some(VerifyProblem::FileMissing {
                path: file.path.clone(),
            }));
        };
        if meta.len() != file.size {
            return Ok(Some(VerifyProblem::SizeMismatch {
                path: file.path.clone(),
                expected: file.size,
                found: meta.len(),
            }));
        }
        if depth == VerifyDepth::Full {
            let found = sha256_hex_file(&path)?;
            if found != file.sha256 {
                return Ok(Some(VerifyProblem::HashMismatch {
                    path: file.path.clone(),
                    expected: file.sha256.clone(),
                    found,
                }));
            }
        }
    }
    Ok(None)
}

/// The full check [`crate::availability`] runs against a resolved path
/// that lives under `user_models_dir()`: the sidecar must exist, parse,
/// and name the locked revision, and every locked file must check out via
/// [`verify_files`].
///
/// See [`verify_files`] for the `Ok`/`Err` split.
pub(crate) fn verify_dir(
    dir: &Path,
    sidecar_path: &Path,
    stt: &SttLock,
    depth: VerifyDepth,
) -> io::Result<Option<VerifyProblem>> {
    let Some(sidecar_bytes) = read_optional(sidecar_path)? else {
        return Ok(Some(VerifyProblem::SidecarMissing));
    };
    let sidecar: SttLock = match serde_json::from_slice(&sidecar_bytes) {
        Ok(parsed) => parsed,
        Err(_) => return Ok(Some(VerifyProblem::SidecarUnparsable)),
    };
    if sidecar.revision != stt.revision {
        return Ok(Some(VerifyProblem::RevisionMismatch {
            expected: stt.revision.clone(),
            found: sidecar.revision,
        }));
    }
    verify_files(dir, stt, depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vuho_model_paths::LockedFile;

    fn sample_lock() -> SttLock {
        SttLock {
            dir_name: "sample-model".to_owned(),
            revision: "deadbeef".to_owned(),
            total_bytes: 11,
            files: vec![
                LockedFile {
                    path: "a.bin".to_owned(),
                    size: 5,
                    sha256: sha256_hex_of(b"hello"),
                },
                LockedFile {
                    path: "sub/b.bin".to_owned(),
                    size: 6,
                    sha256: sha256_hex_of(b"world!"),
                },
            ],
        }
    }

    fn sha256_hex_of(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex::encode(hasher.finalize())
    }

    fn write_matching_tree(dir: &Path, lock: &SttLock) {
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::write(dir.join("a.bin"), b"hello").unwrap();
        fs::write(dir.join("sub/b.bin"), b"world!").unwrap();
        fs::write(sidecar_path(dir), sidecar_bytes(lock)).unwrap();
    }

    #[test]
    fn quick_verify_passes_on_matching_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let lock = sample_lock();
        let model_dir = tmp.path().join(&lock.dir_name);
        write_matching_tree(&model_dir, &lock);
        let sidecar = sidecar_path(&model_dir);

        let result = verify_dir(&model_dir, &sidecar, &lock, VerifyDepth::Quick).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn full_verify_passes_on_matching_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let lock = sample_lock();
        let model_dir = tmp.path().join(&lock.dir_name);
        write_matching_tree(&model_dir, &lock);
        let sidecar = sidecar_path(&model_dir);

        let result = verify_dir(&model_dir, &sidecar, &lock, VerifyDepth::Full).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn missing_sidecar_is_a_problem() {
        let tmp = tempfile::tempdir().unwrap();
        let lock = sample_lock();
        let model_dir = tmp.path().join(&lock.dir_name);
        fs::create_dir_all(&model_dir).unwrap();
        let sidecar = sidecar_path(&model_dir);

        let result = verify_dir(&model_dir, &sidecar, &lock, VerifyDepth::Quick).unwrap();
        assert_eq!(result, Some(VerifyProblem::SidecarMissing));
    }

    #[test]
    fn truncated_file_is_a_size_mismatch_at_quick_depth() {
        let tmp = tempfile::tempdir().unwrap();
        let lock = sample_lock();
        let model_dir = tmp.path().join(&lock.dir_name);
        write_matching_tree(&model_dir, &lock);
        fs::write(model_dir.join("a.bin"), b"he").unwrap(); // truncated
        let sidecar = sidecar_path(&model_dir);

        let result = verify_dir(&model_dir, &sidecar, &lock, VerifyDepth::Quick).unwrap();
        assert!(matches!(result, Some(VerifyProblem::SizeMismatch { .. })));
    }

    #[test]
    fn corrupted_same_size_file_only_caught_at_full_depth() {
        let tmp = tempfile::tempdir().unwrap();
        let lock = sample_lock();
        let model_dir = tmp.path().join(&lock.dir_name);
        write_matching_tree(&model_dir, &lock);
        fs::write(model_dir.join("a.bin"), b"HELLO").unwrap(); // same size, wrong bytes
        let sidecar = sidecar_path(&model_dir);

        let quick = verify_dir(&model_dir, &sidecar, &lock, VerifyDepth::Quick).unwrap();
        assert_eq!(
            quick, None,
            "Quick depth never hashes, so a same-size corruption passes"
        );

        let full = verify_dir(&model_dir, &sidecar, &lock, VerifyDepth::Full).unwrap();
        assert!(matches!(full, Some(VerifyProblem::HashMismatch { .. })));
    }

    #[test]
    fn revision_mismatch_is_a_problem() {
        let tmp = tempfile::tempdir().unwrap();
        let lock = sample_lock();
        let model_dir = tmp.path().join(&lock.dir_name);
        write_matching_tree(&model_dir, &lock);
        let sidecar_path_buf = sidecar_path(&model_dir);
        let mut stale = sample_lock();
        stale.revision = "stale-revision".to_owned();
        fs::write(&sidecar_path_buf, sidecar_bytes(&stale)).unwrap();

        let result = verify_dir(&model_dir, &sidecar_path_buf, &lock, VerifyDepth::Quick).unwrap();
        assert!(matches!(
            result,
            Some(VerifyProblem::RevisionMismatch { .. })
        ));
    }

    #[test]
    fn sidecar_round_trips_through_atomic_write() {
        let tmp = tempfile::tempdir().unwrap();
        let lock = sample_lock();
        let path = sidecar_path(tmp.path());

        vuho_model_paths::atomic_write(&path, &sidecar_bytes(&lock)).unwrap();
        let read_back: SttLock = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();

        assert_eq!(read_back.dir_name, lock.dir_name);
        assert_eq!(read_back.revision, lock.revision);
        assert_eq!(read_back.total_bytes, lock.total_bytes);
        assert_eq!(read_back.files.len(), lock.files.len());
        for (a, b) in read_back.files.iter().zip(lock.files.iter()) {
            assert_eq!(a.path, b.path);
            assert_eq!(a.size, b.size);
            assert_eq!(a.sha256, b.sha256);
        }
    }
}
