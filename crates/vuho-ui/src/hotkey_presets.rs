//! Mapping from persisted [`HotkeySetting`] presets to the concrete
//! [`HotkeyConfig`] the `CGEventTap`-based [`HotkeyListener`] understands.
//!
//! Lives in `vuho-ui` (the composition root, which already depends on both
//! `vuho-settings` and `vuho-os-integration`) so that `vuho-settings` stays
//! serde-only (no `CGEventFlags`, which isn't serializable) and
//! `vuho-os-integration` stays settings-free.
//!
//! [`HotkeyListener`]: vuho_os_integration::HotkeyListener

use objc2_core_graphics::CGEventFlags;
use vuho_os_integration::HotkeyConfig;
use vuho_settings::HotkeySetting;

/// Virtual keycode for the Space key (`kVK_Space`).
const KEYCODE_SPACE: u16 = 49;
/// Virtual keycode for the D key (`kVK_ANSI_D`).
const KEYCODE_D: u16 = 2;

/// Map a persisted hotkey preset to the concrete `HotkeyConfig` the
/// `CGEventTap`-based listener understands.
#[must_use]
pub(crate) fn to_hotkey_config(preset: HotkeySetting) -> HotkeyConfig {
    match preset {
        HotkeySetting::CapsLock => HotkeyConfig::CapsLock,
        HotkeySetting::OptionSpace => HotkeyConfig::Chord {
            flags: CGEventFlags::MaskAlternate,
            keycode: KEYCODE_SPACE,
        },
        HotkeySetting::ControlOptionSpace => HotkeyConfig::Chord {
            flags: CGEventFlags::MaskControl | CGEventFlags::MaskAlternate,
            keycode: KEYCODE_SPACE,
        },
        HotkeySetting::CommandShiftSpace => HotkeyConfig::Chord {
            flags: CGEventFlags::MaskCommand | CGEventFlags::MaskShift,
            keycode: KEYCODE_SPACE,
        },
        HotkeySetting::ControlOptionD => HotkeyConfig::Chord {
            flags: CGEventFlags::MaskControl | CGEventFlags::MaskAlternate,
            keycode: KEYCODE_D,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unwrap a `Chord` variant, panicking with a clear message otherwise.
    fn expect_chord(config: &HotkeyConfig) -> (CGEventFlags, u16) {
        match *config {
            HotkeyConfig::Chord { flags, keycode } => (flags, keycode),
            HotkeyConfig::CapsLock => panic!("expected Chord, got CapsLock"),
        }
    }

    #[test]
    fn caps_lock_maps_to_caps_lock() {
        assert!(matches!(
            to_hotkey_config(HotkeySetting::CapsLock),
            HotkeyConfig::CapsLock
        ));
    }

    #[test]
    fn option_space_maps_to_alternate_space_chord() {
        let (flags, keycode) = expect_chord(&to_hotkey_config(HotkeySetting::OptionSpace));
        assert_eq!(flags, CGEventFlags::MaskAlternate);
        assert_eq!(keycode, KEYCODE_SPACE);
    }

    #[test]
    fn control_option_space_maps_to_combined_chord() {
        let (flags, keycode) = expect_chord(&to_hotkey_config(HotkeySetting::ControlOptionSpace));
        assert_eq!(
            flags,
            CGEventFlags::MaskControl | CGEventFlags::MaskAlternate
        );
        assert_eq!(keycode, KEYCODE_SPACE);
    }

    #[test]
    fn command_shift_space_maps_to_combined_chord() {
        let (flags, keycode) = expect_chord(&to_hotkey_config(HotkeySetting::CommandShiftSpace));
        assert_eq!(flags, CGEventFlags::MaskCommand | CGEventFlags::MaskShift);
        assert_eq!(keycode, KEYCODE_SPACE);
    }

    #[test]
    fn control_option_d_maps_to_combined_chord() {
        let (flags, keycode) = expect_chord(&to_hotkey_config(HotkeySetting::ControlOptionD));
        assert_eq!(
            flags,
            CGEventFlags::MaskControl | CGEventFlags::MaskAlternate
        );
        assert_eq!(keycode, KEYCODE_D);
    }

    /// Every preset must be exercised above — this guards against a
    /// forgotten match arm silently defaulting via a wildcard pattern.
    #[test]
    fn every_preset_is_covered() {
        assert_eq!(HotkeySetting::ALL.len(), 5);
        for preset in HotkeySetting::ALL {
            let _ = to_hotkey_config(preset);
        }
    }
}
