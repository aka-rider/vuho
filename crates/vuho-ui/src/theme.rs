//! The single visual-language chokepoint for the `vuho-ui` crate
//! (CONSTITUTION rule 26): every color, radius, and type-scale value shared
//! across the overlay, settings window, and readiness window lives here, so
//! restyling the app means editing this file, not hunting down scattered
//! literals.
//!
//! Values are derived from — not invented over — the UI's existing look:
//! `overlay.rs`'s opacity-first white text scale (`TEXT_*`), the action-blue
//! and relaunch-green already used for buttons in `readiness.rs`
//! (`ACCENT`/`OK_GREEN`), and the overlay's recording-LED red (`ERROR_RED`).
//! `WARN_AMBER` is the one genuinely new semantic color, reserved for a
//! warning status this crate doesn't yet render.
//!
//! This WP (theme.rs + the overlay restyle that consumes it) only wires up
//! `overlay.rs`; `settings_window.rs`/`readiness.rs` are restyled by other,
//! parallel work packages against these same frozen names. Until they land,
//! several items below (`ACCENT`, `OK_GREEN`, `WARN_AMBER`, most `FILL_*`,
//! `RADIUS_CARD`, `RADIUS_CONTROL`, `TEXT_XS`, `TEXT_MD`, `section_card`,
//! `section_label`, `progress_bar`) have no call site yet — hence the
//! crate-wide `dead_code` allow below, rather than trimming the API those
//! other work packages are already coded against.
#![allow(dead_code)]

use gpui::{div, prelude::*, px, Div, Hsla, SharedString};

// ── Text (opacity-first whites — the overlay's established principle) ─────

/// Confirmed transcript text / outcome headlines: the most prominent text on
/// a dark panel.
pub(crate) const TEXT_PRIMARY: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: 0.95,
};
/// Secondary text: outcome/status messages.
pub(crate) const TEXT_SECONDARY: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: 0.85,
};
/// Tertiary text: captions, section labels, descriptions.
pub(crate) const TEXT_TERTIARY: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: 0.60,
};
/// Disabled/de-emphasized text: unconfirmed transcript tail, idle state.
pub(crate) const TEXT_DISABLED: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: 0.40,
};

// ── Semantic (status only, never decoration) ───────────────────────────────

/// Action-button blue.
pub(crate) const ACCENT: Hsla = Hsla {
    h: 0.55,
    s: 0.5,
    l: 0.45,
    a: 1.0,
};
/// Success/confirmation green (the readiness window's "Relaunch Vuho").
pub(crate) const OK_GREEN: Hsla = Hsla {
    h: 0.35,
    s: 0.5,
    l: 0.4,
    a: 1.0,
};
/// Warning amber.
pub(crate) const WARN_AMBER: Hsla = Hsla {
    h: 0.12,
    s: 0.65,
    l: 0.55,
    a: 1.0,
};
/// The overlay's recording-LED red — the only saturated hue on the panel
/// (see `overlay.rs`'s palette note). Base alpha is opaque; callers that
/// need a dimmer red (the LED's glow, its idle-adjacent states) scale it
/// with [`Hsla::opacity`].
pub(crate) const ERROR_RED: Hsla = Hsla {
    h: 0.0,
    s: 0.75,
    l: 0.55,
    a: 1.0,
};

// ── Fills (white-alpha) ─────────────────────────────────────────────────────

pub(crate) const FILL_CARD: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: 0.06,
};
pub(crate) const FILL_CONTROL: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: 0.08,
};
pub(crate) const FILL_HOVER: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: 0.06,
};
pub(crate) const FILL_SELECTED: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: 0.10,
};
pub(crate) const FILL_DISABLED: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: 0.05,
};
pub(crate) const SEPARATOR: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: 0.08,
};

// ── Radii (px) ──────────────────────────────────────────────────────────────

pub(crate) const RADIUS_PANEL: f32 = 16.0;
pub(crate) const RADIUS_CARD: f32 = 8.0;
pub(crate) const RADIUS_CONTROL: f32 = 6.0;
pub(crate) const RADIUS_CHIP: f32 = 4.0;

// ── Type scale (px) ─────────────────────────────────────────────────────────

/// Captions.
pub(crate) const TEXT_XS: f32 = 11.0;
/// Secondary text / section labels.
pub(crate) const TEXT_SM: f32 = 12.0;
/// Controls / body text.
pub(crate) const TEXT_MD: f32 = 13.0;
/// Headline / transcript text.
pub(crate) const TEXT_LG: f32 = 16.0;

// ── Helpers ──────────────────────────────────────────────────────────────

/// A padded, rounded card surface — the base container for a grouped
/// section (e.g. one readiness-window requirement row).
pub(crate) fn section_card() -> Div {
    div().p_3().rounded(px(RADIUS_CARD)).bg(FILL_CARD)
}

/// A small section header above a control — [`TEXT_SM`], [`TEXT_TERTIARY`],
/// matching `settings_window.rs`'s existing section labels.
pub(crate) fn section_label(text: impl Into<SharedString>) -> Div {
    div()
        .text_size(px(TEXT_SM))
        .text_color(TEXT_TERTIARY)
        .child(text.into())
}

/// A horizontal progress bar: a [`FILL_CONTROL`] track with an [`ACCENT`]
/// fill sized to `fraction` (clamped to `0.0..=1.0`).
pub(crate) fn progress_bar(fraction: f32) -> Div {
    let fraction = fraction.clamp(0.0, 1.0);
    div()
        .h(px(6.0))
        .w_full()
        .rounded(px(RADIUS_CHIP))
        .bg(FILL_CONTROL)
        .child(
            div()
                .h_full()
                .w(gpui::relative(fraction))
                .rounded(px(RADIUS_CHIP))
                .bg(ACCENT),
        )
}
