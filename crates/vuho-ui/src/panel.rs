//! The unified panel (ARCHITECTURE.md ADR-021): one non-activating,
//! always-on-top `NSPanel` showing a single tabbed presentation (Overlay /
//! Settings) — there is no separate click-through dictation-overlay
//! presentation any more. The window is always mouse-interactive but never
//! takes keyboard focus (`window_config::apply_window_config`'s
//! `setBecomesKeyOnlyIfNeeded`), so clicking the tab strip or a Settings
//! control never steals the destination of `inject_text`'s synthesized ⌘V
//! away from whatever app the user is dictating into.
//!
//! **Height follows the active [`Tab`], not the entry point** ([`panel_height`]):
//! `Tab::Overlay` is short (just the tab strip + the dictation content),
//! `Tab::Settings` is tall (permission rows, dropdowns, the speech-model
//! card). The panel's bottom edge is invariant ([`PANEL_BOTTOM_MARGIN`],
//! [`bottom_center_origin`]) — switching tabs grows or shrinks the window
//! *upward*, never moving the bottom edge, so the content always fills the
//! frame exactly and the `AppKit` window shadow always matches the visible
//! panel (no unpainted-but-shadowed region).
//!
//! Caps lock (a session starting) and the tray icon show the **identical**
//! panel; the entry point only decides which tab is initially active
//! (`show`/`on_session_started`/`show_if_hidden`).
//!
//! Compiled under both `--features demo` and production: the demo build's
//! [`PanelRoot`] has no `StatusModel`/`SettingsTab`/permission machinery, so
//! its own `Render` impl just wraps the overlay content in [`panel_chrome`]
//! with no tab strip — there is no `Tab::Settings` to switch to (its
//! variant still exists, `#[allow(dead_code)]`'d there, only so the type is
//! shared between both builds).

use gpui::{
    div, prelude::*, px, App, Bounds, Context, Div, Entity, Hsla, IntoElement, Pixels, Point,
    Render, Size, Window, WindowBounds, WindowHandle, WindowKind, WindowOptions,
};

use crate::overlay::OverlayModel;
use crate::theme;
use crate::window_config;

#[cfg(not(feature = "demo"))]
use std::time::Duration;

#[cfg(not(feature = "demo"))]
use gpui::AsyncApp;

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
use vuho_domain::ModelStatus;

/// Distance from the bottom of the display to the panel's bottom edge.
/// Unchanged from the old overlay window's `OVERLAY_BOTTOM_MARGIN` — the
/// invariant every tab's frame is anchored to (see the module doc comment).
const PANEL_BOTTOM_MARGIN: Pixels = px(120.0);

const PANEL_WIDTH: Pixels = px(460.0);

/// Height of the Overlay tab: the 36px tab strip
/// ([`TAB_STRIP_HEIGHT`]) plus the 180px the dictation content occupies
/// (unchanged from the old standalone overlay window's height).
const OVERLAY_TAB_HEIGHT: Pixels = px(216.0);
/// Height of the Settings tab: permission rows, dropdowns, and the speech
/// model card routinely exceed 420px even before a download is in progress;
/// the tab body scrolls (`PanelRoot::render_tab_body`), but the taller
/// default keeps a first-launch scroll less likely. Referenced from
/// [`panel_height`] in both builds (`Tab::Settings` exists in both — see
/// that enum's doc comment — so the match must stay exhaustive even though
/// a demo build never actually constructs that variant).
const SETTINGS_TAB_HEIGHT: Pixels = px(480.0);

/// Poll interval for re-checking permissions while the panel is open on the
/// Settings tab. Matches the old readiness window's `GATE_POLL_INTERVAL` —
/// frequent enough to feel live, cheap enough not to matter (three
/// synchronous TCC reads, no I/O).
#[cfg(not(feature = "demo"))]
const PERMISSIONS_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Height of the tab strip.
#[cfg(not(feature = "demo"))]
const TAB_STRIP_HEIGHT: Pixels = px(36.0);
/// Icon size within a tab button.
#[cfg(not(feature = "demo"))]
const TAB_ICON_SIZE: Pixels = px(16.0);
/// Tab label text size.
#[cfg(not(feature = "demo"))]
const TAB_LABEL_SIZE: Pixels = px(13.0);

/// The panel's one background opacity (F20/G-collapse): near-opaque, since
/// this is a surface the user reads and clicks (permission rows, dropdowns,
/// buttons, the live transcript) rather than a click-through glance-only
/// overlay — desktop content showing through form controls would cost
/// legibility. Same hue/saturation/lightness as before
/// (`theme::PANEL_HUE`/`PANEL_SATURATION`/`PANEL_LIGHTNESS`, F20).
const PANEL_BG_OPACITY: f32 = 0.95;
/// The panel's border opacity — unchanged from the old Hud chrome's own
/// `PANEL_BORDER_OPACITY`, now the panel's only border.
const PANEL_BORDER_OPACITY: f32 = 0.1;

/// The panel's background color — the single home for this value
/// (CONSTITUTION rule 26): [`panel_chrome`] paints it, and
/// `overlay.rs`'s `fade_strip` fades to it so the transcript viewport's
/// top edge melts into exactly the same surface it sits on.
pub(crate) const PANEL_BG: Hsla = Hsla {
    h: theme::PANEL_HUE,
    s: theme::PANEL_SATURATION,
    l: theme::PANEL_LIGHTNESS,
    a: PANEL_BG_OPACITY,
};
const PANEL_BORDER: Hsla = Hsla {
    h: 0.0,
    s: 0.0,
    l: 1.0,
    a: PANEL_BORDER_OPACITY,
};

/// Which tab the panel currently shows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tab {
    Overlay,
    // A demo build's `PanelRoot` never switches tabs at all (see the module
    // doc comment) — narrow allow, not a file-wide one (rule 29).
    #[cfg_attr(feature = "demo", allow(dead_code))]
    Settings,
}

/// This tab's frame height ([`panel_bounds`]'s one input besides the
/// display). The single place tab → height is decided (CONSTITUTION rule
/// 26).
fn panel_height(tab: Tab) -> Pixels {
    match tab {
        Tab::Overlay => OVERLAY_TAB_HEIGHT,
        Tab::Settings => SETTINGS_TAB_HEIGHT,
    }
}

/// The panel's root view.
pub(crate) struct PanelRoot {
    active_tab: Tab,
    /// Whether the window is currently ordered in (visible on screen).
    shown: bool,
    /// Whether the panel's current visibility was triggered automatically
    /// (a dictation session starting/finishing — [`show_if_hidden`]) rather
    /// than an explicit user open ([`show`]). [`hide_after_session`] only
    /// closes the panel while this is `true` — a finished session must
    /// never close a panel the user opened deliberately.
    auto_shown: bool,
    /// Spawned by [`show`] (guarded against a duplicate spawn — see its
    /// doc comment), self-terminating inside [`start_permissions_poll`]'s
    /// own loop once `!shown && permissions_missing.is_empty()` (G1) —
    /// **not** bounded to the panel's own visibility. Invariant: whenever
    /// `StatusModel::permissions_missing` is non-empty, a poll must keep
    /// running until it observes that list empty, regardless of whether the
    /// panel is currently shown — otherwise a permission granted after the
    /// user dismisses the panel would leave the tray/menu stuck on
    /// `CompositeStatus::PermissionsMissing` forever, since nothing else
    /// ever re-derives that field. [`hide_root`] therefore only clears this
    /// field early when `permissions_missing` is already empty; otherwise
    /// it leaves the task running past dismissal, to be cleaned up by its
    /// own self-termination once permissions converge. `None` under
    /// `--features demo`, which never reaches [`Tab::Settings`].
    #[cfg(not(feature = "demo"))]
    permissions_poll: Option<gpui::Task<()>>,
    pub(crate) overlay: Entity<OverlayModel>,
    #[cfg(not(feature = "demo"))]
    pub(crate) settings: Entity<SettingsTab>,
    #[cfg(not(feature = "demo"))]
    pub(crate) status: Entity<StatusModel>,
}

// ── Construction ─────────────────────────────────────────────────────────

/// Shared `WindowOptions` for both builds: `PopUp` (non-activating panel),
/// no titlebar, created hidden, transparent background (the panel's own
/// chrome — [`panel_chrome`] — paints its own background).
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

/// Create the panel window (hidden initially, `Tab::Overlay`), anchored
/// bottom-center — production build: wires the shared `StatusModel`/
/// `SettingsTab` entities the caller (`main.rs`) already built.
#[cfg(not(feature = "demo"))]
pub(crate) fn create_panel(
    status: Entity<StatusModel>,
    settings: Entity<SettingsTab>,
    cx: &mut App,
) -> WindowHandle<PanelRoot> {
    let bounds = panel_bounds(cx, Tab::Overlay);
    cx.open_window(panel_window_options(bounds), move |window, cx| {
        window.set_window_title("Vuho");
        window_config::apply_window_config(window);
        let overlay = cx.new(|cx| OverlayModel::new(window, cx));
        cx.new(|cx| {
            cx.observe(&overlay, |_this, _overlay, cx| cx.notify())
                .detach();
            cx.observe(&settings, |_this, _settings, cx| cx.notify())
                .detach();
            cx.observe(&status, |_this, _status, cx| cx.notify())
                .detach();
            PanelRoot {
                active_tab: Tab::Overlay,
                shown: false,
                auto_shown: false,
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
    let bounds = panel_bounds(cx, Tab::Overlay);
    cx.open_window(panel_window_options(bounds), |window, cx| {
        window.set_window_title("Vuho");
        window_config::apply_window_config(window);
        let overlay = cx.new(|cx| OverlayModel::new(window, cx));
        cx.new(|cx| {
            cx.observe(&overlay, |_this, _overlay, cx| cx.notify())
                .detach();
            PanelRoot {
                active_tab: Tab::Overlay,
                shown: false,
                auto_shown: false,
                overlay,
            }
        })
    })
    .expect("failed to create panel window")
}

// ── Geometry (pure, unit-tested) ───────────────────────────────────────────

fn panel_size(tab: Tab) -> Size<Pixels> {
    Size {
        width: PANEL_WIDTH,
        height: panel_height(tab),
    }
}

/// Compute the panel's top-left origin so it sits horizontally centered and
/// `bottom_margin` above the bottom edge of the given display bounds —
/// clamped so a display too short to hold `size` above `bottom_margin` pins
/// the panel to the display's top edge instead of pushing it off-screen.
///
/// Pure helper (unit-tested); [`panel_bounds`] handles the display lookup.
fn bottom_center_origin(
    display: Bounds<Pixels>,
    size: Size<Pixels>,
    bottom_margin: Pixels,
) -> Point<Pixels> {
    Point {
        x: display.origin.x + (display.size.width - size.width) * 0.5,
        y: (display.origin.y + display.size.height - size.height - bottom_margin)
            .max(display.origin.y),
    }
}

/// `tab`'s frame: bottom-center on the primary display, at the origin if
/// there is no display at all (matching `WindowBounds::centered`'s
/// no-display fallback in spirit — this crate needs the plain
/// `Bounds<Pixels>` shape for [`window_config::set_frame`], not the
/// `WindowBounds` enum).
///
/// The single source of the panel's geometry: window creation and every
/// [`apply_geometry`] call resolve through here, so no tab, and no
/// transition, can place the window anywhere else. Both tabs share the same
/// bottom edge (`origin.y + size.height`) by construction — only `size`
/// varies with `tab`; `bottom_center_origin` always re-derives `origin.y`
/// from the same `bottom_margin`.
fn panel_bounds(cx: &App, tab: Tab) -> Bounds<Pixels> {
    let size = panel_size(tab);
    match cx.primary_display() {
        Some(display) => Bounds {
            origin: bottom_center_origin(display.bounds(), size, PANEL_BOTTOM_MARGIN),
            size,
        },
        None => Bounds {
            origin: Point {
                x: px(0.0),
                y: px(0.0),
            },
            size,
        },
    }
}

// ── Geometry surgery (the one chokepoint — CONSTITUTION rule 26) ──────────

/// Apply the `NSWindow` frame implied by `tab`. The **only** place a tab
/// switch's window-level effect is applied — every public transition
/// function below goes through this, and `render_tab_button`'s click
/// listener calls it directly whenever the active tab changes.
///
/// Re-resolves the frame against whichever display is primary *now*, so the
/// panel follows a display change (a monitor unplugged, a new primary chosen
/// in System Settings) that happened since the last time it was shown. See
/// `window_config::set_frame`'s G2 note for why that resolution is always
/// against the primary display.
fn apply_geometry(cx: &App, tab: Tab) {
    window_config::set_frame(panel_bounds(cx, tab));
}

// ── Public transitions ─────────────────────────────────────────────────────

/// Show the panel on `tab`, re-fronting it (never taking key status — see
/// the module doc comment) and marking the open as explicit (`auto_shown =
/// false`, so [`hide_after_session`] never closes it). Seeds
/// `StatusModel::permissions_missing` synchronously (G7) so the Settings
/// tab's very first paint is already truthful, then starts the permissions
/// poll (guarded against a duplicate: a no-op if one is already running —
/// G1), and, when opening on the Settings tab, refreshes its device list (a
/// device plugged in since the tab was last shown should already be there).
#[cfg(not(feature = "demo"))]
pub(crate) fn show(panel: WindowHandle<PanelRoot>, tab: Tab, cx: &mut App) {
    let needs_poll = panel
        .update(cx, |root, _window, cx| {
            root.active_tab = tab;
            root.auto_shown = false;
            apply_geometry(cx, tab);
            window_config::order_front();
            root.shown = true;
            if tab == Tab::Settings {
                root.settings.update(cx, SettingsTab::refresh_devices);
            }
            // G7: seed synchronously, through the same derivation the poll
            // itself uses, so the Settings tab's first paint (right here,
            // via `cx.notify()` below) is never one frame stale — the
            // poll's own cx.spawn'd task hasn't had a chance to run its
            // first tick yet at this point.
            refresh_permissions_missing(&root.status, cx);
            cx.notify();
            root.permissions_poll.is_none()
        })
        .unwrap_or(false);
    if needs_poll {
        let task = start_permissions_poll(panel, cx);
        let _ = panel.update(cx, |root, _window, _cx| {
            root.permissions_poll = Some(task);
        });
    }
}

/// The tray icon's plain-click / "Open Vuho" action: hide the panel if it's
/// already showing (F1 — the standard menu-bar-app toggle: clicking an
/// already-open app's icon closes it) — otherwise show it on the Overlay tab
/// while a session is actually live (so a click during dictation never
/// buries the transcript behind whatever tab was last open), or on whichever
/// tab was last active.
#[cfg(not(feature = "demo"))]
pub(crate) fn open_from_tray(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let next = panel.update(cx, |root, _window, cx| {
        if root.shown {
            None
        } else if root.overlay.read(cx).has_session_content() {
            Some(Tab::Overlay)
        } else {
            Some(root.active_tab)
        }
    });
    match next {
        Ok(Some(tab)) => show(panel, tab, cx),
        Ok(None) => hide(panel, cx),
        Err(_) => {}
    }
}

/// Show the panel on `tab` if it isn't currently visible at all — the
/// shared "make sure something is on screen" step behind both
/// [`on_session_started`] (a session actually beginning) and G4's outcome
/// re-show (a `SessionCompleted` outcome that still needs the user's
/// attention, arriving after the panel was already dismissed). No-op while
/// the panel is already shown — never steals focus from a window already on
/// screen. Marks the show as automatic (`auto_shown = true`), so
/// [`hide_after_session`] is allowed to close it again once the session
/// ends.
fn show_if_hidden_root(root: &mut PanelRoot, cx: &App, tab: Tab) {
    if root.shown {
        return;
    }
    root.active_tab = tab;
    apply_geometry(cx, tab);
    window_config::order_front();
    root.shown = true;
    root.auto_shown = true;
}

/// `WindowHandle`-based entry point for [`show_if_hidden_root`] — G4's
/// public re-show call (`event_loop`'s `maybe_show_for_outcome`).
#[cfg(not(feature = "demo"))]
pub(crate) fn show_if_hidden(panel: WindowHandle<PanelRoot>, cx: &mut App, tab: Tab) {
    let _ = panel.update(cx, |root, _window, cx| {
        show_if_hidden_root(root, cx, tab);
        cx.notify();
    });
}

/// React to a dictation session actually starting (`event_loop`'s
/// `SessionStarted`/show-worthy-`Error` handling): show the panel on the
/// Overlay tab if it wasn't visible at all ([`show_if_hidden_root`]), and
/// switch a currently-open panel back to the Overlay tab (shrinking it back
/// down if it was on Settings) so the live transcript is never hidden
/// behind Settings.
pub(crate) fn on_session_started(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let _ = panel.update(cx, |root, _window, cx| {
        show_if_hidden_root(root, cx, Tab::Overlay);
        root.active_tab = Tab::Overlay;
        apply_geometry(cx, Tab::Overlay);
        cx.notify();
    });
}

/// Whether the panel window is currently visible. `false` if the window is
/// gone. `event_loop`'s `ModelStatus::Failed` handling uses this to decide
/// whether a `Failed` status should surface the panel — routine ticks must
/// never do so, but a `Failed` one should still not steal focus from a
/// panel the user already has open.
#[cfg(not(feature = "demo"))]
pub(crate) fn is_shown(panel: WindowHandle<PanelRoot>, cx: &mut App) -> bool {
    panel
        .update(cx, |root, _window, _cx| root.shown)
        .unwrap_or(false)
}

/// Hide the panel — but only while it was shown automatically
/// ([`PanelRoot::auto_shown`]); a no-op while the user opened it explicitly
/// (nothing about a finished dictation session should close a window the
/// user opened deliberately).
pub(crate) fn hide_after_session(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let _ = panel.update(cx, |root, _window, cx| {
        if !root.auto_shown {
            return;
        }
        window_config::order_out();
        root.shown = false;
        // permissions_poll is deliberately left untouched here (F4, amended
        // G1): reaching this point already means the panel was only
        // automatically shown, and the only two ways to get there are (a)
        // the panel was never explicitly opened at all — the poll, which
        // only [`show`] ever starts, was never spawned — or (b) [`hide`]
        // already ran (`hide_root`). In case (b), G1 means `hide_root` may
        // have deliberately left a still-running poll alive (permissions
        // were still missing at dismissal time) rather than clearing the
        // field — see `permissions_poll`'s own doc comment for the
        // invariant this preserves. Either way, this function has nothing
        // new to decide: it's not the one responsible for the field either
        // at rest (`None`) or while a poll is legitimately still
        // converging.
        cx.notify();
    });
}

/// Hide the panel unconditionally (F1) — the tab strip's "✕" button, Esc,
/// and re-clicking an already-open tray icon all go through this, unlike
/// [`hide_after_session`] (which only a finished dictation session's
/// auto-hide calls, and which deliberately leaves an explicitly-opened
/// panel alone).
#[cfg(not(feature = "demo"))]
pub(crate) fn hide(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let _ = panel.update(cx, |root, _window, cx| {
        hide_root(root, cx);
        cx.notify();
    });
}

/// The shared body of "unconditionally close the panel": order the window
/// out, close any open Settings dropdowns (G6 — a stale mic/hotkey list
/// left open across a dismiss/reopen is never useful), drop the permissions
/// poll only if it has nothing left to converge on (G1 — see
/// `permissions_poll`'s own doc comment for the invariant a still-missing
/// poll must keep running past dismissal), and clear `auto_shown` (the panel
/// is no longer showing at all, automatically or otherwise). `active_tab`
/// is deliberately left as-is — what tab a reopened panel lands on is
/// [`open_from_tray`]'s decision (session-content check first), not
/// something closing needs to reset. The window's frame is left as-is too:
/// unlike the old Hud/Full split, there is nothing to reset it *to* — the
/// next show recomputes it from whichever tab that show targets.
///
/// The one implementation both [`hide`] (the public `WindowHandle`-based
/// entry point used by `event_loop`/the tray/Esc) and the tab strip's close
/// button (already inside a `PanelRoot` update, with no need to re-enter
/// through a `WindowHandle`) call, so there is exactly one hide
/// implementation, not two.
#[cfg(not(feature = "demo"))]
fn hide_root(root: &mut PanelRoot, cx: &mut App) {
    window_config::order_out();
    root.shown = false;
    root.auto_shown = false;
    if root.status.read(cx).permissions_missing.is_empty() {
        root.permissions_poll = None;
    }
    root.settings
        .update(cx, |settings, _cx| settings.close_dropdowns());
}

// ── Permissions poll ─────────────────────────────────────────────────────

/// Refresh `StatusModel::permissions_missing` from
/// [`readiness::missing_permissions`], writing only when the value actually
/// changed (so a granted-permission tick that changes nothing never
/// triggers a spurious repaint) — returns whether the freshly-read list is
/// empty. The single derivation every writer goes through (G7/F6):
/// [`show`]'s synchronous first-paint seed and
/// [`start_permissions_poll`]'s own tick both call this and can therefore
/// never disagree in shape (`main.rs`'s `run_gate_blocked` seed is the one
/// remaining writer that predates the panel existing at all and so can't
/// call it — see that function's own doc comment).
#[cfg(not(feature = "demo"))]
fn refresh_permissions_missing(status: &Entity<StatusModel>, cx: &mut App) -> bool {
    let missing = readiness::missing_permissions();
    let is_empty = missing.is_empty();
    status.update(cx, |status, cx| {
        if status.permissions_missing != missing {
            status.permissions_missing = missing;
            cx.notify();
        }
    });
    is_empty
}

/// Re-check permissions immediately (via [`refresh_permissions_missing`]),
/// then every [`PERMISSIONS_POLL_INTERVAL`] thereafter. Sanctioned
/// pull-only TCC polling (no wall-clock ordering of events).
///
/// G1: self-terminates once `!shown && missing.is_empty()` — **not**
/// bounded to the panel's own visibility. A poll spawned while the panel
/// was open must keep running even after the user dismisses the panel, for
/// as long as permissions are still missing: only this loop ever clears
/// `StatusModel::permissions_missing`, so a poll that stopped the moment the
/// panel was hidden would leave the tray/menu stuck reporting
/// `CompositeStatus::PermissionsMissing` forever once the grant actually
/// lands with nobody watching for it. [`show`] guards against a duplicate
/// spawn (only starts a new poll when `permissions_poll` is `None`);
/// [`hide_root`] mirrors this loop's own termination condition, only
/// clearing the field early when permissions are already granted.
///
/// The immediate first tick (F4) runs *before* the first wait, not after —
/// without it, `StatusModel::permissions_missing` would sit on whatever
/// `show`'s own G7 seed left it at for a full [`PERMISSIONS_POLL_INTERVAL`]
/// before this task's first write.
#[cfg(not(feature = "demo"))]
fn start_permissions_poll(panel: WindowHandle<PanelRoot>, cx: &mut App) -> gpui::Task<()> {
    cx.spawn(move |cx: &mut AsyncApp| {
        let mut cx = cx.clone();
        async move {
            loop {
                let result = panel.update(&mut cx, |root, _window, cx| {
                    let is_empty = refresh_permissions_missing(&root.status, cx);
                    (root.shown, is_empty)
                });
                let Ok((shown, is_empty)) = result else {
                    log::info!("panel: permissions poll stopping — panel window gone");
                    return;
                };
                if !shown && is_empty {
                    log::info!(
                        "panel: permissions poll stopping — panel hidden and permissions granted"
                    );
                    return;
                }
                cx.background_executor()
                    .timer(PERMISSIONS_POLL_INTERVAL)
                    .await;
            }
        }
    })
}

// ── Rendering ────────────────────────────────────────────────────────────

/// The panel's one chrome: background, border, radius, shadow. Both
/// `Render` impls build on this — there is no second chrome anywhere (rule
/// 26); `overlay.rs`'s `fade_strip` fades to the same [`PANEL_BG`] rather
/// than painting its own.
fn panel_chrome() -> Div {
    div()
        .size_full()
        .flex()
        .flex_col()
        .bg(PANEL_BG)
        .border_1()
        .border_color(PANEL_BORDER)
        .rounded(px(theme::RADIUS_PANEL))
        .shadow_lg()
        .text_color(theme::TEXT_PRIMARY)
}

#[cfg(feature = "demo")]
impl Render for PanelRoot {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // A demo build never shows a tab strip — see the module doc
        // comment — so the render is unconditionally just the overlay
        // content inside the shared chrome.
        panel_chrome().child(self.overlay.read(cx).render_content())
    }
}

#[cfg(not(feature = "demo"))]
impl Render for PanelRoot {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.render_panel(cx)
    }
}

#[cfg(not(feature = "demo"))]
impl PanelRoot {
    /// The panel's outer chrome ([`panel_chrome`]) + tab strip + active
    /// tab's body.
    fn render_panel(&self, cx: &mut Context<Self>) -> impl IntoElement {
        panel_chrome()
            .relative()
            .child(self.render_tab_strip(cx))
            .child(self.render_tab_body(cx))
    }

    /// The 36px tab strip: Overlay / Settings, a trailing spacer, the "✕"
    /// close button (F1), bottom `SEPARATOR` hairline.
    fn render_tab_strip(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .h(TAB_STRIP_HEIGHT)
            .px_2()
            .gap_1()
            .border_b_1()
            .border_color(theme::SEPARATOR)
            .child(self.render_tab_button(
                Tab::Overlay,
                WAVEFORM_ICON,
                "Overlay",
                "panel-tab-overlay",
                cx,
            ))
            .child(self.render_tab_button(
                Tab::Settings,
                GEAR_ICON,
                "Settings",
                "panel-tab-settings",
                cx,
            ))
            .child(div().flex_1())
            .child(Self::render_close_button(cx))
    }

    /// The "✕" close button at the tab strip's right end (F1) — hides the
    /// panel via [`hide_root`] (the same implementation [`hide`] itself
    /// calls). 24×24px hit target, `TEXT_TERTIARY` at rest, `FILL_HOVER` +
    /// `TEXT_PRIMARY` on hover — mirrors the tab buttons' own hover
    /// treatment, `RADIUS_CONTROL` rounding. An associated function, not a
    /// method: it renders purely from `cx`'s listener plumbing, touching no
    /// `&self` field.
    fn render_close_button(cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("panel-close")
            .flex()
            .items_center()
            .justify_center()
            .size(px(24.0))
            .rounded(px(theme::RADIUS_CONTROL))
            .cursor_pointer()
            .text_color(theme::TEXT_TERTIARY)
            .hover(|style| style.bg(theme::FILL_HOVER).text_color(theme::TEXT_PRIMARY))
            .child("✕")
            .on_click(cx.listener(|this, _event, _window, cx| {
                hide_root(this, cx);
                cx.notify();
            }))
    }

    /// One tab button: icon + label, a `FILL_SELECTED` pill when active,
    /// `TEXT_TERTIARY`/`FILL_HOVER` otherwise. Clicking sets `active_tab`,
    /// re-applies the frame for the newly-active tab
    /// ([`apply_geometry`] — the resize trigger for a tab switch), and
    /// refreshes the microphone device list when switching to Settings
    /// (same "don't show a stale device list" reasoning as [`show`]'s own
    /// refresh).
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
            apply_geometry(cx, tab);
            if tab == Tab::Settings {
                this.settings.update(cx, SettingsTab::refresh_devices);
            }
            cx.notify();
        }))
    }

    /// The active tab's body, filling the remaining space below the strip.
    ///
    /// `.p_4()` here is the Settings tab's own padding chokepoint (G5:
    /// previously shared with the Overlay tab too, which double-padded
    /// whenever a session was live — `OverlayModel::render_content` already
    /// carries its own `px_6`/`py_4` inset). The Overlay tab now insets
    /// nowhere here; [`Self::render_overlay_tab`] applies padding only to
    /// its idle-status branch, which has none of its own.
    ///
    /// The Settings tab additionally scrolls (F2 — its content routinely
    /// overflows [`SETTINGS_TAB_HEIGHT`]): `.id(..)` + `.overflow_y_scroll()`
    /// is gpui 0.2's stateful-scroll idiom (`StatefulInteractiveElement`,
    /// vendored `gpui-0.2.2/src/elements/div.rs`) for the simple case that
    /// needs no explicit `ScrollHandle`. The Overlay tab stays
    /// `overflow_hidden` instead — its idle block/live transcript are
    /// already height-bounded, and scrolling live transcript would be
    /// actively wrong.
    fn render_tab_body(&self, cx: &mut Context<Self>) -> impl IntoElement {
        match self.active_tab {
            Tab::Overlay => div()
                .flex_1()
                .overflow_hidden()
                .child(self.render_overlay_tab(cx))
                .into_any_element(),
            Tab::Settings => div()
                .id("panel-settings-scroll")
                .flex_1()
                .p_4()
                .overflow_y_scroll()
                .child(self.settings.clone())
                .into_any_element(),
        }
    }

    /// The Overlay tab's body: the live overlay content while a session is
    /// actually in progress or its outcome is still flashing, otherwise an
    /// idle status block driven by `StatusModel`.
    ///
    /// G5: the two branches pad differently, deliberately.
    /// `OverlayModel::render_content` already carries its own `px_6`/`py_4`
    /// inset, so wrapping it in another `.p_4()` here would double-pad it;
    /// [`Self::render_idle_status`] has no inset of its own, so this is the
    /// one place that supplies it.
    fn render_overlay_tab(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        if self.overlay.read(cx).has_session_content() {
            return self.overlay.read(cx).render_content();
        }
        div()
            .size_full()
            .p_4()
            .child(self.render_idle_status(cx))
            .into_any_element()
    }

    /// The idle status block: `StatusModel::idle_headline`'s headline/
    /// sub-line, a progress bar while `Downloading`, and — only for the
    /// three states a click here can actually resolve — an "Open Settings"
    /// button that switches to the Settings tab. Never a disabled dead
    /// button: the button simply doesn't render for every other state.
    ///
    /// No outer padding of its own (F16, amended G5) —
    /// [`Self::render_overlay_tab`]'s idle branch is the padding chokepoint
    /// this inherits from now.
    fn render_idle_status(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let status = self.status.read(cx);
        let (headline, sub) = status.idle_headline();
        let composite = status.composite();

        let mut column = div().flex().flex_col().gap_2().child(
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

        // F11: read `status.model`'s own `fraction()` directly — the same
        // derivation `settings_tab.rs`'s Speech Model card uses — instead of
        // re-deriving a fraction from the rounded `u8` percent the tray/menu
        // display, which loses precision `progress_bar`'s width doesn't need
        // to lose.
        if matches!(composite, CompositeStatus::Downloading(_)) {
            let fraction = status
                .model
                .as_ref()
                .and_then(ModelStatus::fraction)
                .unwrap_or(0.0);
            column = column.child(theme::progress_bar(fraction));
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
                    apply_geometry(cx, Tab::Settings);
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
        let size = panel_size(Tab::Overlay);
        let origin = bottom_center_origin(display, size, px(120.0));
        // Horizontally centered: (1000 - 460) / 2 = 270.
        assert_eq!(origin.x, px(270.0));
        // Bottom-anchored: 800 - 216 - 120 = 464.
        assert_eq!(origin.y, px(464.0));
    }

    /// The real invariant a tab-driven resize must never break: switching
    /// tabs changes `size.height` but must never move the bottom edge
    /// (`origin.y + size.height`).
    #[test]
    fn both_tabs_share_a_bottom_edge() {
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
        let overlay_origin =
            bottom_center_origin(display, panel_size(Tab::Overlay), PANEL_BOTTOM_MARGIN);
        let settings_origin =
            bottom_center_origin(display, panel_size(Tab::Settings), PANEL_BOTTOM_MARGIN);
        let overlay_bottom = overlay_origin.y + panel_height(Tab::Overlay);
        let settings_bottom = settings_origin.y + panel_height(Tab::Settings);
        assert_eq!(overlay_bottom, settings_bottom);
    }

    /// `Tab::Settings` is the taller tab (permission rows, dropdowns, the
    /// speech-model card).
    #[test]
    fn settings_tab_is_taller_than_overlay_tab() {
        assert!(panel_height(Tab::Settings) > panel_height(Tab::Overlay));
    }

    /// A display too short to fit the frame above the bottom margin pins the
    /// panel to the display's top edge rather than pushing it off-screen.
    #[test]
    fn bottom_center_origin_clamps_to_short_display() {
        let display = Bounds {
            origin: Point {
                x: px(0.0),
                y: px(0.0),
            },
            size: Size {
                width: px(1024.0),
                height: px(200.0),
            },
        };
        let origin = bottom_center_origin(display, panel_size(Tab::Settings), PANEL_BOTTOM_MARGIN);
        // Unclamped this would be 200 - 480 - 120 = -400, off the top edge.
        assert_eq!(origin.y, px(0.0));
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
