//! Shared hand-rolled widgets for the Settings tab (`settings_tab.rs`).
//!
//! Copied — not moved — from `settings_window.rs`'s dropdown trio
//! (`dropdown_button`/`dropdown_option_list`/`dropdown_option`, ~286-331)
//! and `readiness.rs`'s button styling (`action_button`/`button_base`), then
//! restyled to consume `theme.rs`'s visual-language consts instead of the
//! ad hoc `hsla(..)` literals those two modules still carry. The originals
//! are left untouched: they die wholesale in a later integration package
//! that replaces both windows with the unified panel this module's only
//! caller (`settings_tab.rs`) belongs to — copying now avoids a
//! half-migrated shared dependency between old and new UI in the meantime.
//!
//! Only what `settings_tab.rs` actually renders lives here — no speculative
//! extras.

// TODO(ui-rehaul): remove once wired — `settings_tab.rs` is this module's
// only caller, and neither is reachable from `main.rs` yet (the later
// integration package embeds the Settings tab in the unified panel).
#![allow(dead_code)]

use gpui::{
    div, prelude::*, px, App, ClickEvent, Div, ElementId, Hsla, IntoElement, SharedString, Window,
};

use crate::theme;

/// The always-visible dropdown button showing the current value, with a
/// trailing disclosure glyph. Click toggles the caller's own open/closed
/// state (this widget is stateless).
///
/// `label_color` is caller-supplied (not hardcoded to [`theme::TEXT_PRIMARY`])
/// because `settings_tab.rs`'s microphone row needs to render the current
/// value in [`theme::WARN_AMBER`] when the persisted device is disconnected
/// (see `settings_tab::mic_display`).
pub(crate) fn dropdown_button(
    current: impl Into<SharedString>,
    label_color: Hsla,
    id: impl Into<ElementId>,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .id(id.into())
        .flex()
        .justify_between()
        .items_center()
        .px_3()
        .py_2()
        .rounded(px(theme::RADIUS_CONTROL))
        .bg(theme::FILL_CONTROL)
        .cursor_pointer()
        .child(div().text_color(label_color).child(current.into()))
        .child("▾")
        .on_click(on_click)
}

/// Container for a dropdown's expanded option list.
pub(crate) fn dropdown_option_list() -> Div {
    div()
        .flex()
        .flex_col()
        .mt_1()
        .rounded(px(theme::RADIUS_CONTROL))
        .bg(theme::FILL_SELECTED)
        .overflow_hidden()
}

/// One selectable row within an expanded dropdown. `text_color` lets
/// `settings_tab.rs` grey out a persisted-but-disconnected microphone entry
/// ([`theme::TEXT_DISABLED`]) while keeping it clickable.
pub(crate) fn dropdown_option(
    text: impl Into<SharedString>,
    text_color: Hsla,
    id: impl Into<ElementId>,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .id(id.into())
        .px_3()
        .py_2()
        .cursor_pointer()
        .hover(|style| style.bg(theme::FILL_HOVER))
        .child(div().text_color(text_color).child(text.into()))
        .on_click(on_click)
}

/// Base style shared by every clickable pill/button below — background
/// color is the only thing that varies per call site (`theme::ACCENT` for
/// the default action, `theme::OK_GREEN` for a relaunch confirmation).
fn button_base(bg: Hsla) -> Div {
    div()
        .px_3()
        .py_2()
        .rounded(px(theme::RADIUS_CONTROL))
        .bg(bg)
        .cursor_pointer()
}

/// A clickable action button (Download/Retry/Allow/Open System
/// Settings/Relaunch, …) — a colored pill with a label and a click handler.
pub(crate) fn action_button(
    label: impl Into<SharedString>,
    id: impl Into<ElementId>,
    bg: Hsla,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    button_base(bg)
        .id(id.into())
        .child(label.into())
        .on_click(on_click)
}

/// A non-interactive status pill — no `cursor_pointer`, no click handler —
/// for a state a click can't resolve (e.g. "In progress…" while a download
/// is already running).
pub(crate) fn disabled_pill(label: impl Into<SharedString>) -> impl IntoElement {
    div()
        .px_3()
        .py_2()
        .rounded(px(theme::RADIUS_CONTROL))
        .bg(theme::FILL_DISABLED)
        .text_color(theme::TEXT_DISABLED)
        .child(label.into())
}
