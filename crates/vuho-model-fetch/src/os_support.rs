//! Whether the running OS can run a given model.
//!
//! Canary's encoder and decoder ship int4 weights, which `CoreML` only
//! executes from macOS 15 — while the project floor is 14.0 — and the ONNX
//! Parakeet export runs only on Linux. What each model needs is declared in
//! `models.manifest.json` as `requires`, and this module is the one place
//! that compares it against the running system, so the Settings UI can
//! withhold a download the machine could never run. (How large that
//! download is comes from `models.lock.json`'s `total_bytes`, which the UI
//! already renders — restating a figure here would be a second copy to keep
//! in sync, CONSTITUTION rule 2.)

use vuho_model_paths::{MacosVersion, Os, Requires};

/// Whether the running OS can run a model, and if not, why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    /// The running OS meets the model's requirement.
    Supported,
    /// The model is for macOS and the running macOS is older than this.
    NeedsMacos(MacosVersion),
    /// The model is built for a different OS.
    OtherOs(Os),
}

impl Support {
    /// Whether the model can run here at all.
    #[must_use]
    pub fn is_supported(self) -> bool {
        self == Self::Supported
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn support(requires: &Requires) -> Support {
    use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo};

    match *requires {
        Requires::Linux => Support::OtherOs(Os::Linux),
        Requires::Macos { min_version } => {
            let version = NSOperatingSystemVersion {
                majorVersion: min_version.major as isize,
                minorVersion: min_version.minor as isize,
                patchVersion: 0,
            };
            if NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(version) {
                Support::Supported
            } else {
                Support::NeedsMacos(min_version)
            }
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn support(requires: &Requires) -> Support {
    match requires {
        Requires::Linux => Support::Supported,
        Requires::Macos { .. } => Support::OtherOs(Os::Macos),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn macos(major: u32, minor: u32) -> Requires {
        Requires::Macos {
            min_version: MacosVersion { major, minor },
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_linux_model_is_supported_on_linux() {
        assert_eq!(support(&Requires::Linux), Support::Supported);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_macos_model_is_for_another_os_on_linux() {
        assert_eq!(support(&macos(14, 0)), Support::OtherOs(Os::Macos));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_linux_model_is_for_another_os_on_macos() {
        assert_eq!(support(&Requires::Linux), Support::OtherOs(Os::Linux));
    }

    /// Every macOS this crate can even build for is ≥ 14.0 (the workspace's
    /// declared floor), so a floor below that must always be satisfied —
    /// which also proves the `NSProcessInfo` call itself works.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_floor_below_the_project_minimum_is_supported_on_macos() {
        assert_eq!(support(&macos(10, 0)), Support::Supported);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_floor_above_any_released_macos_names_that_floor() {
        assert_eq!(
            support(&macos(999, 0)),
            Support::NeedsMacos(MacosVersion {
                major: 999,
                minor: 0
            })
        );
    }

    #[test]
    fn only_supported_is_supported() {
        assert!(Support::Supported.is_supported());
        assert!(!Support::OtherOs(Os::Linux).is_supported());
        assert!(!Support::NeedsMacos(MacosVersion {
            major: 15,
            minor: 0
        })
        .is_supported());
    }
}
