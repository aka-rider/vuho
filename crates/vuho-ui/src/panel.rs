//! The unified panel (ARCHITECTURE.md ADR-021): one non-activating,
//! always-on-top `NSPanel` with two presentations —
//!
//! - [`Presentation::Hud`]: the dictation overlay, unchanged from the old
//!   standalone overlay window — bottom-center, click-through, shown on
//!   `SessionStarted`, auto-hidden per outcome.
//! - [`Presentation::Full`]: a centered, opaque, tabbed window (Overlay /
//!   Settings) that replaces the old lazy settings window *and* the
//!   permission/model readiness window — the Settings tab shows permission
//!   rows and speech-model provisioning exactly like the old gate window
//!   did, so opening the panel on the Settings tab at launch *is* the gate.
//!
//! Compiled under both `--features demo` and production: the demo build's
//! [`PanelRoot`] never leaves [`Presentation::Hud`] (there is no
//! `StatusModel`/`SettingsTab`/permission machinery to show a Full
//! presentation of), so its own `Render` impl and [`create_panel`] overload
//! are much smaller than production's.

use gpui::{
    prelude::*, px, App, Bounds, Context, Entity, IntoElement, Pixels, Point, Render, Size,
    Window, WindowBounds, WindowHandle, WindowKind, WindowOptions,
};

use crate::overlay::OverlayModel;
use crate::window_config;

#[cfg(not(feature = "demo"))]
use std::time::Duration;

#[cfg(not(feature = "demo"))]
use gpui::{div, AsyncApp, ParentElement, Styled};

#[cfg(not(feature = "demo"))]
use crate::app_status::{CompositeStatus, StatusModel};
#[cfg(not(feature = "demo"))]
use crate::assets::{GEAR_ICON, WAVEFORM_ICON};
#[cfg(not(feature = "demo"))]
use crate::controls;
#[cfg(not(feature = "demo"))]
use crate::readiness;
#[cfg(not(feature = "demo"))]
use crate::settings_tab::SettingsTab;
#[cfg(not(feature = "demo"))]
use crate::theme;

/// Distance from the bottom of the display to the Hud's bottom edge.
/// Unchanged from the old overlay window's `OVERLAY_BOTTOM_MARGIN`.
const HUD_BOTTOM_MARGIN: Pixels = px(120.0);

/// Hud presentation dimensions — unchanged from the old overlay window's
/// `OVERLAY_WIDTH`/`OVERLAY_HEIGHT`.
const HUD_WIDTH: Pixels = px(460.0);
const HUD_HEIGHT: Pixels = px(180.0);

/// Full presentation dimensions: ~460×420, centered on the primary display.
const FULL_WIDTH: Pixels = px(460.0);
const FULL_HEIGHT: Pixels = px(420.0);

/// Poll interval for re-checking permissions while the panel is open on the
/// Full presentation. Matches the old readiness window's `GATE_POLL_INTERVAL`
/// — frequent enough to feel live, cheap enough not to matter (three
/// synchronous TCC reads, no I/O).
#[cfg(not(feature = "demo"))]
const PERMISSIONS_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Height of the Full presentation's tab strip.
#[cfg(not(feature = "demo"))]
const TAB_STRIP_HEIGHT: Pixels = px(36.0);
/// Icon size within a tab button.
#[cfg(not(feature = "demo"))]
const TAB_ICON_SIZE: Pixels = px(16.0);
/// Tab label text size.
#[cfg(not(feature = "demo"))]
const TAB_LABEL_SIZE: Pixels = px(13.0);

/// Same hue/saturation/lightness as the Hud's translucent chrome
/// (`overlay.rs`'s `PANEL_HUE`/`PANEL_SATURATION`/`PANEL_LIGHTNESS`), but
/// opaque: the Full presentation is a real, focusable window surface, not a
/// floating overlay, so content behind it must never show through.
#[cfg(not(feature = "demo"))]
const FULL_BG: gpui::Hsla = gpui::Hsla {
    h: 0.7,
    s: 0.1,
    l: 0.08,
    a: 0.97,
};

/// Which shape the panel currently renders as.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Presentation {
    /// The dictation overlay: bottom-center, click-through, no keyboard
    /// focus.
    Hud,
    /// The tabbed Overlay/Settings window: centered, opaque, focusable.
    Full,
}

/// Which tab the Full presentation currently shows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tab {
    Overlay,
    // Never leaves `Presentation::Hud` under `--features demo` (see the
    // module doc comment), so nothing in a demo build ever constructs this
    // variant — narrow allow, not a file-wide one (rule 29).
    #[cfg_attr(feature = "demo", allow(dead_code))]
    Settings,
}

/// The panel's root view. See the module doc comment for the two
/// presentations this renders.
pub(crate) struct PanelRoot {
    presentation: Presentation,
    active_tab: Tab,
    /// Whether the window is currently ordered in (visible on screen).
    shown: bool,
    /// Spawned by [`show_full`], dropped by [`hide_if_hud`] — see
    /// `start_permissions_poll`'s doc comment. `None` under `--features
    /// demo`, which never reaches [`Presentation::Full`].
    #[cfg(not(feature = "demo"))]
    permissions_poll: Option<gpui::Task<()>>,
    pub(crate) overlay: Entity<OverlayModel>,
    #[cfg(not(feature = "demo"))]
    pub(crate) settings: Entity<SettingsTab>,
    #[cfg(not(feature = "demo"))]
    pub(crate) status: Entity<StatusModel>,
}

// ── Construction ─────────────────────────────────────────────────────────

/// Shared `WindowOptions` for both builds — unchanged from the old overlay
/// window: `PopUp` (non-activating panel that can become key), no titlebar,
/// created hidden, transparent background (the panel's own chrome elements
/// paint their own background — see [`crate::overlay::hud_chrome`] and
/// [`PanelRoot::render_full`]).
fn panel_window_options(bounds: Bounds<Pixels>) -> WindowOptions {
    WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(bounds)),
        titlebar: None,
        focus: false,
        show: false,
        kind: WindowKind::PopUp,
        is_resizable: false,
        is_minimizable: false,
        window_background: gpui::WindowBackgroundAppearance::Transparent,
        ..Default::default()
    }
}

/// Create the panel window (hidden initially, [`Presentation::Hud`]),
/// anchored bottom-center — production build: wires the shared
/// `StatusModel`/`SettingsTab` entities the caller (`main.rs`) already
/// built.
#[cfg(not(feature = "demo"))]
pub(crate) fn create_panel(
    status: Entity<StatusModel>,
    settings: Entity<SettingsTab>,
    cx: &mut App,
) -> WindowHandle<PanelRoot> {
    let bounds = hud_bounds(cx);
    cx.open_window(panel_window_options(bounds), move |window, cx| {
        window.set_window_title("Vuho");
        window_config::apply_window_config(window);
        let overlay = cx.new(|cx| OverlayModel::new(window, cx));
        cx.new(|cx| {
            cx.observe(&overlay, |_this, _overlay, cx| cx.notify()).detach();
            cx.observe(&settings, |_this, _settings, cx| cx.notify()).detach();
            cx.observe(&status, |_this, _status, cx| cx.notify()).detach();
            PanelRoot {
                presentation: Presentation::Hud,
                active_tab: Tab::Overlay,
                shown: false,
                permissions_poll: None,
                overlay,
                settings,
                status,
            }
        })
    })
    .expect("failed to create panel window")
}

/// Create the panel window — demo build: just the overlay entity, no
/// `StatusModel`/`SettingsTab` (the demo has no menu bar, no settings, no
/// permissions to show).
#[cfg(feature = "demo")]
pub(crate) fn create_panel(cx: &mut App) -> WindowHandle<PanelRoot> {
    let bounds = hud_bounds(cx);
    cx.open_window(panel_window_options(bounds), |window, cx| {
        window.set_window_title("Vuho");
        window_config::apply_window_config(window);
        let overlay = cx.new(|cx| OverlayModel::new(window, cx));
        cx.new(|cx| {
            cx.observe(&overlay, |_this, _overlay, cx| cx.notify()).detach();
            PanelRoot {
                presentation: Presentation::Hud,
                active_tab: Tab::Overlay,
                shown: false,
                overlay,
            }
        })
    })
    .expect("failed to create panel window")
}

// ── Geometry (pure, unit-tested) ───────────────────────────────────────────

fn hud_size() -> Size<Pixels> {
    Size {
        width: HUD_WIDTH,
        height: HUD_HEIGHT,
    }
}

fn full_size() -> Size<Pixels> {
    Size {
        width: FULL_WIDTH,
        height: FULL_HEIGHT,
    }
}

/// Compute the Hud's top-left origin so it sits horizontally centered and
/// `bottom_margin` above the bottom edge of the given display bounds.
///
/// Pure helper (unit-tested); [`hud_bounds`] handles the display lookup.
/// Moved verbatim from the old `main.rs::bottom_center_origin`.
fn bottom_center_origin(
    display: Bounds<Pixels>,
    size: Size<Pixels>,
    bottom_margin: Pixels,
) -> Point<Pixels> {
    Point {
        x: display.origin.x + (display.size.width - size.width) * 0.5,
        y: display.origin.y + display.size.height - size.height - bottom_margin,
    }
}

/// `size` centered on the primary display, or at the origin if there is no
/// display (matches `WindowBounds::centered`'s no-display fallback in
/// spirit — this crate needs the plain `Bounds<Pixels>` shape for
/// [`window_config::set_frame`], not the `WindowBounds` enum).
fn centered_on_primary(size: Size<Pixels>, cx: &App) -> Bounds<Pixels> {
    match cx.primary_display() {
        Some(display) => {
            let display_bounds = display.bounds();
            let origin = Point {
                x: display_bounds.origin.x + (display_bounds.size.width - size.width) * 0.5,
                y: display_bounds.origin.y + (display_bounds.size.height - size.height) * 0.5,
            };
            Bounds { origin, size }
        }
        None => Bounds {
            origin: Point {
                x: px(0.0),
                y: px(0.0),
            },
            size,
        },
    }
}

/// Bottom-center bounds on the primary display; falls back to screen-centered
/// when no display is available. Used both for the panel's initial creation
/// bounds and for the Hud arm of [`apply_presentation`].
fn hud_bounds(cx: &App) -> Bounds<Pixels> {
    let size = hud_size();
    match cx.primary_display() {
        Some(display) => Bounds {
            origin: bottom_center_origin(display.bounds(), size, HUD_BOTTOM_MARGIN),
            size,
        },
        None => centered_on_primary(size, cx),
    }
}

/// Centered bounds for the Full presentation. Compiled under both features
/// even though only production ever constructs `Presentation::Full` — see
/// [`apply_presentation`], whose `match` must stay exhaustive over
/// [`Presentation`] regardless of `--features demo`.
fn full_bounds(cx: &App) -> Bounds<Pixels> {
    centered_on_primary(full_size(), cx)
}

// ── Presentation surgery (the one chokepoint — CONSTITUTION rule 26) ──────

/// Apply every `NSWindow` change implied by switching to `presentation`:
/// frame, click-through, and (Full only) key status. The **only** place
/// either transition's window-level effects are applied — every public
/// transition function below goes through this.
fn apply_presentation(window: &mut Window, cx: &App, presentation: Presentation) {
    match presentation {
        Presentation::Hud => {
            window_config::set_frame(window, hud_bounds(cx));
            window_config::set_click_through(window, true);
        }
        Presentation::Full => {
            window_config::set_frame(window, full_bounds(cx));
            window_config::set_click_through(window, false);
            window_config::make_key_and_order_front(window);
        }
    }
}

// ── Public transitions ─────────────────────────────────────────────────────

/// Show the Full presentation on `tab`, re-fronting the panel and giving it
/// key status. Starts the permissions poll and, when opening on the
/// Settings tab, refreshes its device list (a device plugged in since the
/// tab was last shown should already be there).
#[cfg(not(feature = "demo"))]
pub(crate) fn show_full(panel: WindowHandle<PanelRoot>, tab: Tab, cx: &mut App) {
    let _ = panel.update(cx, |root, window, cx| {
        root.presentation = Presentation::Full;
        root.active_tab = tab;
        apply_presentation(window, cx, Presentation::Full);
        root.shown = true;
        if tab == Tab::Settings {
            root.settings.update(cx, SettingsTab::refresh_devices);
        }
        cx.notify();
    });
    let task = start_permissions_poll(panel, cx);
    let _ = panel.update(cx, |root, _window, _cx| {
        root.permissions_poll = Some(task);
    });
}

/// The tray icon's plain-click / "Open Vuho" action: show the Full
/// presentation on the Overlay tab while a session is actually live (so a
/// click during dictation never buries the transcript behind whatever tab
/// was last open), or on whichever tab was last active otherwise.
#[cfg(not(feature = "demo"))]
pub(crate) fn open_from_tray(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let tab = panel
        .update(cx, |root, _window, cx| {
            if root.overlay.read(cx).has_session_content() {
                Tab::Overlay
            } else {
                root.active_tab
            }
        })
        .unwrap_or(Tab::Overlay);
    show_full(panel, tab, cx);
}

/// React to a dictation session actually starting (`event_loop`'s
/// `SessionStarted`/show-worthy-`Error` handling): show the panel as the Hud
/// if it wasn't visible at all, or switch a currently-open Full presentation
/// back to the Overlay tab so the live transcript is never hidden behind
/// Settings — but never re-frame or steal key status from an open Full
/// presentation, which would be jarring mid-click.
pub(crate) fn on_session_started(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let _ = panel.update(cx, |root, window, cx| {
        if !root.shown {
            root.presentation = Presentation::Hud;
            apply_presentation(window, cx, Presentation::Hud);
            window_config::order_front(window);
            root.shown = true;
        } else if root.presentation == Presentation::Full {
            root.active_tab = Tab::Overlay;
        }
        cx.notify();
    });
}

/// Whether the panel window is currently visible (either presentation).
/// `false` if the window is gone. `event_loop`'s `ModelStatus::Failed`
/// handling uses this to decide whether a `Failed` status should surface
/// the panel — routine ticks must never do so, but a `Failed` one should
/// still not steal focus/re-frame a panel the user already has open.
#[cfg(not(feature = "demo"))]
pub(crate) fn is_shown(panel: WindowHandle<PanelRoot>, cx: &mut App) -> bool {
    panel
        .update(cx, |root, _window, _cx| root.shown)
        .unwrap_or(false)
}

/// Hide the panel — but only while it's presenting the Hud; a no-op while
/// the Full presentation is open (nothing about a finished dictation session
/// should close a window the user opened deliberately).
pub(crate) fn hide_if_hud(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let _ = panel.update(cx, |root, window, cx| {
        if root.presentation != Presentation::Hud {
            return;
        }
        window_config::order_out(window);
        root.shown = false;
        #[cfg(not(feature = "demo"))]
        {
            root.permissions_poll = None;
        }
        cx.notify();
    });
}

// ── Permissions poll ─────────────────────────────────────────────────────

/// Re-check [`readiness::missing_permissions`] every
/// [`PERMISSIONS_POLL_INTERVAL`] while the Full presentation is open, and
/// write the result into `StatusModel::permissions_missing` — only when it
/// actually changed, so a granted-permission tick that changes nothing never
/// triggers a spurious repaint. Sanctioned pull-only TCC polling (no
/// wall-clock ordering of events), bounded to the panel's own visibility by
/// [`show_full`] spawning it and [`hide_if_hud`] dropping it — mirrors the
/// old readiness window's `spawn_poll_loop` (studied, then reimplemented
/// minimally against `StatusModel` instead of a bespoke view field).
#[cfg(not(feature = "demo"))]
fn start_permissions_poll(panel: WindowHandle<PanelRoot>, cx: &mut App) -> gpui::Task<()> {
    cx.spawn(move |cx: &mut AsyncApp| {
        let mut cx = cx.clone();
        async move {
            loop {
                cx.background_executor()
                    .timer(PERMISSIONS_POLL_INTERVAL)
                    .await;
                let missing = readiness::missing_permissions();
                let updated = panel.update(&mut cx, |root, _window, cx| {
                    root.status.update(cx, |status, cx| {
                        if status.permissions_missing != missing {
                            status.permissions_missing = missing;
                            cx.notify();
                        }
                    });
                });
                if updated.is_err() {
                    log::info!("panel: permissions poll stopping — panel window gone");
                    return;
                }
            }
        }
    })
}

// ── Rendering ────────────────────────────────────────────────────────────

#[cfg(feature = "demo")]
impl Render for PanelRoot {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Demo mode never leaves Presentation::Hud — see the module doc
        // comment — so the render is unconditionally the Hud content.
        crate::overlay::hud_chrome(self.overlay.read(cx).render_content())
    }
}

#[cfg(not(feature = "demo"))]
impl Render for PanelRoot {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        match self.presentation {
            Presentation::Hud => {
                crate::overlay::hud_chrome(self.overlay.read(cx).render_content())
                    .into_any_element()
            }
            Presentation::Full => self.render_full(cx).into_any_element(),
        }
    }
}

#[cfg(not(feature = "demo"))]
impl PanelRoot {
    /// The Full presentation's outer chrome (opaque, rounded) + tab strip +
    /// active tab's body.
    fn render_full(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .bg(FULL_BG)
            .rounded(px(theme::RADIUS_PANEL))
            .shadow_lg()
            .child(self.render_tab_strip(cx))
            .child(self.render_tab_body(cx))
    }

    /// The 36px tab strip: Overlay / Settings, bottom `SEPARATOR` hairline.
    fn render_tab_strip(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .h(TAB_STRIP_HEIGHT)
            .px_2()
            .gap_1()
            .border_b_1()
            .border_color(theme::SEPARATOR)
            .child(self.render_tab_button(Tab::Overlay, WAVEFORM_ICON, "Overlay", "panel-tab-overlay", cx))
            .child(self.render_tab_button(Tab::Settings, GEAR_ICON, "Settings", "panel-tab-settings", cx))
    }

    /// One tab button: icon + label, a `FILL_SELECTED` pill when active,
    /// `TEXT_TERTIARY`/`FILL_HOVER` otherwise. Clicking sets `active_tab`
    /// (and refreshes the microphone device list when switching to
    /// Settings — same "don't show a stale device list" reasoning as
    /// `show_full`'s own refresh).
    fn render_tab_button(
        &self,
        tab: Tab,
        icon: &'static str,
        label: &'static str,
        id: &'static str,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let selected = self.active_tab == tab;
        let text_color = if selected {
            theme::TEXT_PRIMARY
        } else {
            theme::TEXT_TERTIARY
        };

        let mut button = div()
            .id(id)
            .flex()
            .items_center()
            .gap_1p5()
            .px_2()
            .py_1()
            .rounded(px(theme::RADIUS_CONTROL))
            .cursor_pointer()
            .child(
                gpui::svg()
                    .path(icon)
                    .size(TAB_ICON_SIZE)
                    .text_color(text_color),
            )
            .child(
                div()
                    .text_size(TAB_LABEL_SIZE)
                    .text_color(text_color)
                    .child(label),
            );
        button = if selected {
            button.bg(theme::FILL_SELECTED)
        } else {
            button.hover(|style| style.bg(theme::FILL_HOVER))
        };
        button.on_click(cx.listener(move |this, _event, _window, cx| {
            this.active_tab = tab;
            if tab == Tab::Settings {
                this.settings.update(cx, SettingsTab::refresh_devices);
            }
            cx.notify();
        }))
    }

    /// The active tab's body, filling the remaining space below the strip.
    fn render_tab_body(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex_1()
            .overflow_hidden()
            .child(match self.active_tab {
                Tab::Overlay => self.render_overlay_tab(cx).into_any_element(),
                Tab::Settings => self.settings.clone().into_any_element(),
            })
    }

    /// The Overlay tab's body: the live overlay content while a session is
    /// actually in progress or its outcome is still flashing, otherwise an
    /// idle status block driven by `StatusModel`.
    fn render_overlay_tab(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        if self.overlay.read(cx).has_session_content() {
            return self.overlay.read(cx).render_content();
        }
        self.render_idle_status(cx).into_any_element()
    }

    /// The idle status block: `StatusModel::idle_headline`'s headline/
    /// sub-line, a progress bar while `Downloading`, and — only for the
    /// three states a click here can actually resolve — an "Open Settings"
    /// button that switches to the Settings tab. Never a disabled dead
    /// button: the button simply doesn't render for every other state.
    fn render_idle_status(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let status = self.status.read(cx);
        let (headline, sub) = status.idle_headline();
        let composite = status.composite();

        let mut column = div()
            .flex()
            .flex_col()
            .gap_2()
            .p_6()
            .child(
                div()
                    .text_size(px(theme::TEXT_LG))
                    .text_color(theme::TEXT_PRIMARY)
                    .child(headline),
            );

        if let Some(sub) = sub {
            column = column.child(
                div()
                    .text_size(px(theme::TEXT_SM))
                    .text_color(theme::TEXT_SECONDARY)
                    .child(sub),
            );
        }

        if let CompositeStatus::Downloading(pct) = composite {
            column = column.child(theme::progress_bar(f32::from(pct) / 100.0));
        }

        if matches!(
            composite,
            CompositeStatus::ModelMissing
                | CompositeStatus::EngineFailed
                | CompositeStatus::PermissionsMissing
        ) {
            column = column.child(controls::action_button(
                "Open Settings",
                "panel-idle-open-settings",
                theme::ACCENT,
                cx.listener(|this, _event, _window, cx| {
                    this.active_tab = Tab::Settings;
                    this.settings.update(cx, SettingsTab::refresh_devices);
                    cx.notify();
                }),
            ));
        }

        column
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottom_center_origin_centers_and_anchors_bottom() {
        let display = Bounds {
            origin: Point {
                x: px(0.0),
                y: px(0.0),
            },
            size: Size {
                width: px(1000.0),
                height: px(800.0),
            },
        };
        let size = hud_size();
        let origin = bottom_center_origin(display, size, px(120.0));
        // Horizontally centered: (1000 - 460) / 2 = 270.
        assert_eq!(origin.x, px(270.0));
        // Bottom-anchored: 800 - 180 - 120 = 500.
        assert_eq!(origin.y, px(500.0));
    }

    #[test]
    fn bottom_center_origin_respects_display_offset() {
        // A secondary display offset to the right and down.
        let display = Bounds {
            origin: Point {
                x: px(1000.0),
                y: px(100.0),
            },
            size: Size {
                width: px(800.0),
                height: px(600.0),
            },
        };
        let size = Size {
            width: px(400.0),
            height: px(100.0),
        };
        let origin = bottom_center_origin(display, size, px(50.0));
        // x: 1000 + (800 - 400) / 2 = 1200.
        assert_eq!(origin.x, px(1200.0));
        // y: 100 + 600 - 100 - 50 = 550.
        assert_eq!(origin.y, px(550.0));
    }
}
