//! GPUI [`AssetSource`]: the crate's hand-drawn monochrome icons, embedded
//! into the binary via `include_bytes!` so no runtime filesystem lookup (and
//! no packaging step) is needed to find them — matches how `build.rs`
//! already embeds `packaging/Info.plist` link-time rather than shipping it
//! as a loose file.
//!
//! Always compiled (unlike most of this crate's other new WP5 modules): the
//! demo build (`--features demo`) still creates GPUI windows and will want
//! the same icon set once a later package wires the panel/tray into it, so
//! there is no reason to cfg-gate an asset table itself.

use std::borrow::Cow;

use gpui::{AssetSource, Result, SharedString};

/// Path (relative to this crate's `assets/` directory) of the waveform
/// (dictation) icon — resembles SF Symbols "waveform": rounded vertical bars
/// of varying height.
pub(crate) const WAVEFORM_ICON: &str = "icons/waveform.svg";
/// Path of the settings (gear) icon — resembles SF Symbols "gearshape".
pub(crate) const GEAR_ICON: &str = "icons/gear.svg";

/// The crate's [`AssetSource`]: exactly the two icons above, embedded at
/// compile time. `gpui::svg()` elements render whatever this returns as a
/// monochrome sprite tinted with the current text color (see
/// `gpui::window::Window::paint_svg`'s doc comment) — the SVGs themselves
/// carry no color of their own (see the files' own comments).
pub(crate) struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        match path {
            WAVEFORM_ICON => Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/waveform.svg"
            )))),
            GEAR_ICON => Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/gear.svg"
            )))),
            _ => Ok(None),
        }
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        if path == "icons" {
            Ok(vec![
                SharedString::from(WAVEFORM_ICON),
                SharedString::from(GEAR_ICON),
            ])
        } else {
            Ok(Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_both_known_icons() {
        assert!(Assets.load(WAVEFORM_ICON).unwrap().is_some());
        assert!(Assets.load(GEAR_ICON).unwrap().is_some());
    }

    #[test]
    fn loads_none_for_an_unknown_path() {
        assert!(Assets.load("icons/nonexistent.svg").unwrap().is_none());
    }

    #[test]
    fn lists_the_icons_prefix() {
        let listed = Assets.list("icons").unwrap();
        assert_eq!(listed.len(), 2);
        assert!(listed.contains(&SharedString::from(WAVEFORM_ICON)));
        assert!(listed.contains(&SharedString::from(GEAR_ICON)));
    }

    #[test]
    fn lists_empty_for_an_unknown_prefix() {
        assert!(Assets.list("nope").unwrap().is_empty());
    }
}
