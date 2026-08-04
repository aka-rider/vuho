//! The single visual-language chokepoint for the `vuho-ui` crate
//! (CONSTITUTION rule 26): every color, radius, and type-scale value shared
//! across the panel's chrome (`panel.rs`), overlay content (`overlay.rs`),
//! and the Settings tab (`settings_tab.rs`, `controls.rs`) lives here, so
//! restyling the app means editing this file, not hunting down scattered
//! literals.
//!
//! Values are derived from — not invented over — the UI's existing look:
//! `overlay.rs`'s opacity-first white text scale (`TEXT_*`), the action-blue
//! and relaunch-green already used for buttons (`ACCENT`/`OK_GREEN`), and
//! the overlay's recording-LED red (`ERROR_RED`). `WARN_AMBER` is the one
//! genuinely new semantic color, reserved for a warning status.
//!
//! This module is compiled under both `--features demo` and production —
//! `panel.rs`'s shared chrome, which needs it, is compiled in both — but
//! every item consumed only by production-only modules
//! (`settings_tab.rs`/`controls.rs`/`panel.rs`'s tab-strip code, all
//! `#[cfg(not(feature = "demo"))]`) is therefore genuinely dead code under a
//! demo build. That's a module-level `#[cfg_attr(feature = "demo",
//! allow(dead_code))]` below (CONSTITUTION rule 29: scoped to the demo
//! feature specifically, unlike the file-wide, always-on `allow` this
//! replaces — a production build still catches a genuinely unused item).
//! `TEXT_XS` (11px captions) is the one item with no call site in *either*
//! build yet — the module-level allow doesn't reach it under production, so
//! it carries its own item-level `allow` with its own reason, right above
//! its definition.
#![cfg_attr(
    feature = "demo",
    allow(
        dead_code,
        reason = "every semantic/fill/radius/section-helper token here that's consumed only by \
                  settings_tab.rs/controls.rs/panel.rs's tab-strip code (all \
                  #[cfg(not(feature = \"demo\"))]) is genuinely dead when this module compiles \
                  into the demo build, which has no tab strip to switch to Settings with (see \
                  the module doc comment) — TEXT_XS is the one exception, dead under production \
                  too, and covered by its own item-level allow instead"
    )
)]

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

// ── Shared panel background (F20) ──────────────────────────────────────────
//
// `panel.rs`'s single chrome (`PANEL_BG`) paints this hue/saturation/
// lightness at its own opacity — the three magnitudes live here once
// instead of as a hand-copied literal that could drift on a future
// restyle.
//
// `PANEL_LIGHTNESS` is 0.12, not the near-black 0.08 it started at: the
// panel is meant to read as smoked glass over the desktop, and at 0.08 it
// read as a black hole instead.

pub(crate) const PANEL_HUE: f32 = 0.7;
pub(crate) const PANEL_SATURATION: f32 = 0.1;
pub(crate) const PANEL_LIGHTNESS: f32 = 0.12;

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

/// Captions. No call site in either build yet — see the module doc comment.
#[allow(
    dead_code,
    reason = "reserved for a future caption smaller than TEXT_SM; no call site in either build yet"
)]
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

/// A small section header above a control — [`TEXT_SM`], [`TEXT_TERTIARY`].
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
