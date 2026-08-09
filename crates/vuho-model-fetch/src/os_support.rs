//! Whether the running macOS is new enough for a given model.
//!
//! Canary's encoder and decoder ship int4 weights, which `CoreML` only
//! executes from macOS 15 — while the project floor is 14.0. The floor per
//! model is declared in `models.manifest.json` as `min_macos`, and this
//! module is the one place that compares it against the running system, so
//! the Settings UI can withhold a download the machine could never run.
//! (How large that download is comes from `models.lock.json`'s
//! `total_bytes`, which the UI already renders — restating a figure here
//! would be a second copy to keep in sync, CONSTITUTION rule 2.)

use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo};

/// Whether the running macOS is at least `min`, written `"MAJOR.MINOR"`.
///
/// A string that is not `"MAJOR.MINOR"` is *not* satisfied. Treating an
/// unparsable floor as "supported" would offer a download that then fails
/// inside `CoreML` with a shape/opset error naming nothing the user can act
/// on. Refusing it, and logging the value that was refused, surfaces the
/// manifest bug (CONSTITUTION rule 2) — without the log line the model
/// would simply be absent from the UI with nothing anywhere saying why.
pub(crate) fn min_macos_satisfied(min: &str) -> bool {
    let Some((major, minor)) = parse_min_macos(min) else {
        log::error!(
            "vuho-model-fetch: models.manifest.json declares min_macos {min:?}, which is not \"MAJOR.MINOR\" — treating the model as unsupported on every system"
        );
        return false;
    };
    let version = NSOperatingSystemVersion {
        majorVersion: major,
        minorVersion: minor,
        patchVersion: 0,
    };
    NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(version)
}

/// `"15.0"` → `(15, 0)`; anything else → `None`.
fn parse_min_macos(min: &str) -> Option<(isize, isize)> {
    let (major, minor) = min.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_major_minor_floor() {
        assert_eq!(parse_min_macos("15.0"), Some((15, 0)));
        assert_eq!(parse_min_macos("14.7"), Some((14, 7)));
    }

    #[test]
    fn rejects_anything_that_is_not_major_dot_minor() {
        assert_eq!(parse_min_macos("15"), None);
        assert_eq!(parse_min_macos("15.0.1"), None);
        assert_eq!(parse_min_macos("fifteen.zero"), None);
        assert_eq!(parse_min_macos(""), None);
    }

    #[test]
    fn a_malformed_floor_is_not_satisfied() {
        assert!(!min_macos_satisfied("fifteen"));
        assert!(!min_macos_satisfied(""));
    }

    /// Every macOS this crate can even build for is ≥ 14.0 (the workspace's
    /// declared floor), so a floor below that must always be satisfied —
    /// which also proves the `NSProcessInfo` call itself works.
    #[test]
    fn a_floor_below_the_project_minimum_is_satisfied() {
        assert!(min_macos_satisfied("10.0"));
    }
}
