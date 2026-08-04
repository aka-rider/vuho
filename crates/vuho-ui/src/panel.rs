//! The unified panel (ARCHITECTURE.md ADR-021): one non-activating,
//! always-on-top `NSPanel` occupying **one constant frame**
//! ([`panel_bounds`] — bottom-center on the primary display, never moved and
//! never resized), with two presentations painted into it:
//!
//! - [`Presentation::Hud`]: the dictation overlay — click-through, shown on
//!   `SessionStarted`, auto-hidden per outcome. Paints only the bottom
//!   [`crate::overlay::HUD_CHROME_HEIGHT`] of the frame (see
//!   [`crate::overlay::hud_chrome`]) and leaves the rest transparent, so it
//!   looks and sits exactly like the old standalone overlay window.
//! - [`Presentation::Full`]: a near-opaque, tabbed presentation (Overlay /
//!   Settings) filling the whole frame, replacing the old lazy settings
//!   window *and* the permission/model readiness window — the Settings tab
//!   shows permission rows and speech-model provisioning exactly like the old
//!   gate window did, so opening the panel on the Settings tab at launch *is*
//!   the gate.
//!
//! The two presentations deliberately share one frame: when they had their
//! own (460×180 bottom-center vs. 460×480 screen-centered), starting
//! dictation and then opening the panel moved *and* resized the window under
//! the user, which read as two different windows rather than one.
//!
//! Compiled under both `--features demo` and production: the demo build's
//! [`PanelRoot`] never leaves [`Presentation::Hud`] (there is no
//! `StatusModel`/`SettingsTab`/permission machinery to show a Full
//! presentation of), so its own `Render` impl and [`create_panel`] overload
//! are much smaller than production's.

use gpui::{
    prelude::*, px, App, Bounds, Context, Entity, IntoElement, Pixels, Point, Render, Size, Window,
    WindowBounds, WindowHandle, WindowKind, WindowOptions,
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
#[cfg(not(feature = "demo"))]
use vuho_domain::ModelStatus;

/// Distance from the bottom of the display to the panel's bottom edge.
/// Unchanged from the old overlay window's `OVERLAY_BOTTOM_MARGIN`, so the
/// Hud chrome — which the [`Presentation::Hud`] arm pins to that same bottom
/// edge — sits exactly where it always did.
const PANEL_BOTTOM_MARGIN: Pixels = px(120.0);

/// The panel's one and only frame size, shared by both presentations.
///
/// The height is the Full presentation's requirement (the Settings tab's
/// content routinely exceeds 420px even before a Speech Model card is
/// showing; the tab body scrolls — see [`PanelRoot::render_tab_body`] — but
/// the taller default keeps a first-launch scroll less likely). The Hud
/// needs far less and simply leaves the upper part of the frame unpainted
/// and click-through, rather than resizing the window.
const PANEL_WIDTH: Pixels = px(460.0);
const PANEL_HEIGHT: Pixels = px(480.0);

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
/// (`overlay.rs`'s `color_panel_bg`) — both draw from `theme::PANEL_HUE`/
/// `PANEL_SATURATION`/`PANEL_LIGHTNESS` (F20) — but near-opaque, where the
/// Hud is deliberately see-through: this is a surface the user reads and
/// clicks (permission rows, dropdowns, buttons), and desktop content showing
/// through form controls costs legibility the floating overlay can afford to
/// spend.
#[cfg(not(feature = "demo"))]
const FULL_BG: gpui::Hsla = gpui::Hsla {
    h: theme::PANEL_HUE,
    s: theme::PANEL_SATURATION,
    l: theme::PANEL_LIGHTNESS,
    a: 0.97,
};

/// Which shape the panel currently renders as. Both share one frame — see
/// the module doc comment; a presentation decides what is *painted* into
/// that frame, plus click-through and key status, never where the window is
/// or how big it is.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Presentation {
    /// The dictation overlay: click-through, no keyboard focus, painting
    /// only the bottom of the frame.
    Hud,
    /// The tabbed Overlay/Settings presentation: near-opaque, focusable,
    /// filling the frame.
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
    /// Spawned by [`show_full`] (guarded against a duplicate spawn — see its
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
    /// `--features demo`, which never reaches [`Presentation::Full`].
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
    let bounds = panel_bounds(cx);
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
    let bounds = panel_bounds(cx);
    cx.open_window(panel_window_options(bounds), |window, cx| {
        window.set_window_title("Vuho");
        window_config::apply_window_config(window);
        let overlay = cx.new(|cx| OverlayModel::new(window, cx));
        cx.new(|cx| {
            cx.observe(&overlay, |_this, _overlay, cx| cx.notify())
                .detach();
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

fn panel_size() -> Size<Pixels> {
    Size {
        width: PANEL_WIDTH,
        height: PANEL_HEIGHT,
    }
}

/// Compute the panel's top-left origin so it sits horizontally centered and
/// `bottom_margin` above the bottom edge of the given display bounds —
/// clamped so a display too short to hold `size` above `bottom_margin` pins
/// the panel to the display's top edge instead of pushing it off-screen
/// (the frame reaches `PANEL_HEIGHT + PANEL_BOTTOM_MARGIN` = 600px up from
/// the bottom, which does not fit every external display).
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

/// The panel's frame: bottom-center on the primary display, at the origin if
/// there is no display at all (matching `WindowBounds::centered`'s
/// no-display fallback in spirit — this crate needs the plain
/// `Bounds<Pixels>` shape for [`window_config::set_frame`], not the
/// `WindowBounds` enum).
///
/// The single source of the panel's geometry: window creation and every
/// [`apply_presentation`] call resolve through here, so no presentation, and
/// no transition, can place the window anywhere else.
fn panel_bounds(cx: &App) -> Bounds<Pixels> {
    let size = panel_size();
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

// ── Presentation surgery (the one chokepoint — CONSTITUTION rule 26) ──────

/// Apply every `NSWindow` change implied by switching to `presentation`:
/// click-through, and (Full only) key status. The **only** place either
/// transition's window-level effects are applied — every public transition
/// function below goes through this.
///
/// The frame is re-applied here too, but *outside* the match and identically
/// for both presentations ([`panel_bounds`]): switching presentation must
/// never move or resize the window. The call is not redundant — it
/// re-resolves the frame against whichever display is primary *now*, so the
/// panel follows a display change (a monitor unplugged, a new primary chosen
/// in System Settings) that happened since the last time it was shown. See
/// `window_config::set_frame`'s G2 note for why that resolution is always
/// against the primary display.
///
/// `grab_key` only matters for the [`Presentation::Full`] arm — the Hud arm
/// never takes key status regardless of this parameter, so a Hud-bound
/// caller may pass either value. G3(c): [`show_full`] passes `false` when a
/// session is currently recording, so opening the panel mid-dictation
/// (e.g. a `Failed` model status, or the user clicking the tray icon) plain
/// `order_front`s the window instead of stealing key status — and with it,
/// the destination of `inject_text`'s synthesized ⌘V — from the app the
/// user is dictating into.
fn apply_presentation(cx: &App, presentation: Presentation, grab_key: bool) {
    window_config::set_frame(panel_bounds(cx));
    match presentation {
        Presentation::Hud => {
            window_config::set_click_through(true);
        }
        Presentation::Full => {
            window_config::set_click_through(false);
            if grab_key {
                window_config::make_key_and_order_front();
            } else {
                window_config::order_front();
            }
        }
    }
}

// ── Public transitions ─────────────────────────────────────────────────────

/// Show the Full presentation on `tab`, re-fronting the panel and — unless a
/// session is currently recording (G3(c)) — giving it key status. Seeds
/// `StatusModel::permissions_missing` synchronously (G7) so the Settings
/// tab's very first paint is already truthful, then starts the permissions
/// poll (guarded against a duplicate: a no-op if one is already running —
/// G1), and, when opening on the Settings tab, refreshes its device list (a
/// device plugged in since the tab was last shown should already be there).
#[cfg(not(feature = "demo"))]
pub(crate) fn show_full(panel: WindowHandle<PanelRoot>, tab: Tab, cx: &mut App) {
    let needs_poll = panel
        .update(cx, |root, _window, cx| {
            root.presentation = Presentation::Full;
            root.active_tab = tab;
            let grab_key = !root.overlay.read(cx).is_recording();
            apply_presentation(cx, Presentation::Full, grab_key);
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
/// already showing the Full presentation (F1 — the standard menu-bar-app
/// toggle: clicking an already-open app's icon closes it) — otherwise show
/// the Full presentation on the Overlay tab while a session is actually live
/// (so a click during dictation never buries the transcript behind whatever
/// tab was last open), or on whichever tab was last active.
#[cfg(not(feature = "demo"))]
pub(crate) fn open_from_tray(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let next = panel.update(cx, |root, _window, cx| {
        if root.shown && root.presentation == Presentation::Full {
            None
        } else if root.overlay.read(cx).has_session_content() {
            Some(Tab::Overlay)
        } else {
            Some(root.active_tab)
        }
    });
    match next {
        Ok(Some(tab)) => show_full(panel, tab, cx),
        Ok(None) => hide(panel, cx),
        Err(_) => {}
    }
}

/// Show the panel as the Hud if it isn't currently visible at all — the
/// shared "make sure something is on screen" step behind both
/// [`on_session_started`] (a session actually beginning) and G4's
/// [`show_hud_for_outcome`] (a `SessionCompleted` outcome that still needs
/// the user's attention, arriving after the panel was already dismissed).
/// No-op while the panel is already shown, in either presentation — never
/// steals focus from a window already on screen.
fn show_hud_if_hidden(root: &mut PanelRoot, cx: &App) {
    if root.shown {
        return;
    }
    root.presentation = Presentation::Hud;
    apply_presentation(cx, Presentation::Hud, false);
    window_config::order_front();
    root.shown = true;
}

/// React to a dictation session actually starting (`event_loop`'s
/// `SessionStarted`/show-worthy-`Error` handling): show the panel as the Hud
/// if it wasn't visible at all ([`show_hud_if_hidden`]), or switch a
/// currently-open Full presentation back to the Overlay tab so the live
/// transcript is never hidden behind Settings. G3(a): if that already-open Full presentation
/// happens to be key, it yields key status
/// (`window_config::resign_key_keep_front`) — a session starting means the
/// user is about to dictate into some *other* app, and the panel keeping
/// key status would send `inject_text`'s synthesized ⌘V into itself
/// instead.
pub(crate) fn on_session_started(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let _ = panel.update(cx, |root, _window, cx| {
        let was_shown = root.shown;
        show_hud_if_hidden(root, cx);
        if was_shown && root.presentation == Presentation::Full {
            root.active_tab = Tab::Overlay;
            window_config::resign_key_keep_front();
        }
        cx.notify();
    });
}

/// G4: re-show the panel as the Hud when a `SessionCompleted` whose outcome
/// still needs the user's attention (`InjectionOutcome::ClipboardOnly`/
/// `Failed`) arrives after the panel was already dismissed mid-session —
/// otherwise "Copied to clipboard — ⌘V to paste" or a failure message would
/// flash into a hidden window and never be seen. Shares
/// [`show_hud_if_hidden`] with [`on_session_started`], so it's a no-op while
/// the panel is already shown, in either presentation.
#[cfg(not(feature = "demo"))]
pub(crate) fn show_hud_for_outcome(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let _ = panel.update(cx, |root, _window, cx| {
        show_hud_if_hidden(root, cx);
        cx.notify();
    });
}

/// Whether the panel window is currently visible (either presentation).
/// `false` if the window is gone. `event_loop`'s `ModelStatus::Failed`
/// handling uses this to decide whether a `Failed` status should surface
/// the panel — routine ticks must never do so, but a `Failed` one should
/// still not steal focus from a panel the user already has open.
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
    let _ = panel.update(cx, |root, _window, cx| {
        if root.presentation != Presentation::Hud {
            return;
        }
        window_config::order_out();
        root.shown = false;
        // permissions_poll is deliberately left untouched here (F4, amended
        // G1): reaching this point already means `presentation == Hud`, and
        // the only two ways to get there are (a) the panel never left
        // `Presentation::Hud` at all — the poll, which only [`show_full`]
        // ever starts, was never spawned — or (b) [`hide`] already ran
        // (`hide_root`). In case (b), G1 means `hide_root` may have
        // deliberately left a still-running poll alive (permissions were
        // still missing at dismissal time) rather than clearing the field —
        // see `permissions_poll`'s own doc comment for the invariant this
        // preserves. Either way, this function has nothing new to decide:
        // it's not the one responsible for the field either at rest (`None`)
        // or while a poll is legitimately still converging.
        cx.notify();
    });
}

/// Hide the panel unconditionally, regardless of presentation (F1) — the
/// tab strip's "✕" button, Esc, and re-clicking an already-open tray icon
/// all go through this, unlike [`hide_if_hud`] (which only a finished
/// dictation session's auto-hide calls, and which deliberately leaves an
/// open Full presentation alone).
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
/// poll must keep running past dismissal), and reset to [`Presentation::Hud`]
/// (re-applying Hud surgery — click-through — while hidden, so the next
/// [`on_session_started`] shows a click-through Hud instead of the
/// mouse-capturing state the Full presentation left behind). `active_tab` is deliberately left as-is — what tab a
/// reopened Full presentation lands on is [`open_from_tray`]'s decision
/// (session-content check first), not something closing needs to reset.
///
/// The one implementation both [`hide`] (the public `WindowHandle`-based
/// entry point used by `event_loop`/the tray/Esc) and the tab strip's close
/// button (already inside a `PanelRoot` update, with no need to re-enter
/// through a `WindowHandle`) call, so there is exactly one hide
/// implementation, not two. Takes `cx: &mut App` (not the read-only `&App`
/// this used before G6) because closing the Settings dropdowns goes through
/// `Entity::update`, which requires mutable access.
#[cfg(not(feature = "demo"))]
fn hide_root(root: &mut PanelRoot, cx: &mut App) {
    window_config::order_out();
    root.shown = false;
    if root.status.read(cx).permissions_missing.is_empty() {
        root.permissions_poll = None;
    }
    root.settings
        .update(cx, |settings, _cx| settings.close_dropdowns());
    root.presentation = Presentation::Hud;
    apply_presentation(cx, Presentation::Hud, false);
}

// ── Permissions poll ─────────────────────────────────────────────────────

/// Refresh `StatusModel::permissions_missing` from
/// [`readiness::missing_permissions`], writing only when the value actually
/// changed (so a granted-permission tick that changes nothing never
/// triggers a spurious repaint) — returns whether the freshly-read list is
/// empty. The single derivation every writer goes through (G7/F6):
/// [`show_full`]'s synchronous first-paint seed and
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
/// bounded to the panel's own visibility. A poll spawned while the Full
/// presentation was open must keep running even after the user dismisses
/// the panel, for as long as permissions are still missing: only this loop
/// ever clears `StatusModel::permissions_missing`, so a poll that stopped
/// the moment the panel was hidden would leave the tray/menu stuck
/// reporting `CompositeStatus::PermissionsMissing` forever once the grant
/// actually lands with nobody watching for it. [`show_full`] guards against
/// a duplicate spawn (only starts a new poll when `permissions_poll` is
/// `None`); [`hide_root`] mirrors this loop's own termination condition,
/// only clearing the field early when permissions are already granted.
///
/// The immediate first tick (F4) runs *before* the first wait, not after —
/// without it, `StatusModel::permissions_missing` would sit on whatever
/// `show_full`'s own G7 seed left it at for a full
/// [`PERMISSIONS_POLL_INTERVAL`] before this task's first write.
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
            Presentation::Hud => crate::overlay::hud_chrome(self.overlay.read(cx).render_content())
                .into_any_element(),
            Presentation::Full => self.render_full(cx).into_any_element(),
        }
    }
}

#[cfg(not(feature = "demo"))]
impl PanelRoot {
    /// The Full presentation's outer chrome (opaque, rounded) + tab strip +
    /// active tab's body.
    ///
    /// `.text_color(theme::TEXT_PRIMARY)` here (F3) is the single chokepoint
    /// for the Full presentation's default text color — every descendant
    /// glyph that doesn't set its own `text_color` (gpui's cascading
    /// `text_style_stack`, e.g. the tab strip's "▾"/"✕" glyphs and
    /// `controls::action_button`'s label) inherits it, instead of falling
    /// through to gpui's black default on this dark chrome (CONSTITUTION
    /// rule 26 — one chokepoint, not a per-glyph patch at every call site).
    fn render_full(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .relative()
            .size_full()
            .flex()
            .flex_col()
            .bg(FULL_BG)
            .text_color(theme::TEXT_PRIMARY)
            .rounded(px(theme::RADIUS_PANEL))
            .shadow_lg()
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
    ///
    /// `.p_4()` here is the Settings tab's own padding chokepoint (G5:
    /// previously shared with the Overlay tab too, which double-padded
    /// whenever a session was live — `OverlayModel::render_content` already
    /// carries its own `px_6`/`py_4` inset, since it's also embedded
    /// unpadded in the Hud presentation's chrome). The Overlay tab now
    /// insets nowhere here; [`Self::render_overlay_tab`] applies padding
    /// only to its idle-status branch, which has none of its own.
    ///
    /// The Settings tab additionally scrolls (F2 — its content routinely
    /// overflows [`PANEL_HEIGHT`]): `.id(..)` + `.overflow_y_scroll()` is
    /// gpui 0.2's stateful-scroll idiom (`StatefulInteractiveElement`,
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
    /// inset (shared with the Hud presentation, which embeds it unpadded —
    /// see that method's doc comment), so wrapping it in another `.p_4()`
    /// here would double-pad it; [`Self::render_idle_status`] has no inset
    /// of its own, so this is the one place that supplies it.
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
    /// this inherits from now (`render_tab_body`'s own `.p_4()` applies to
    /// the Settings tab only, since G5 found live overlay content
    /// double-padded under the old shared chokepoint); this used to carry
    /// its own `.p_6()`, diverging from the Settings tab's padding.
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
        let size = panel_size();
        let origin = bottom_center_origin(display, size, px(120.0));
        // Horizontally centered: (1000 - 460) / 2 = 270.
        assert_eq!(origin.x, px(270.0));
        // Bottom-anchored: 800 - 480 - 120 = 200.
        assert_eq!(origin.y, px(200.0));
    }

    /// The Hud paints a band pinned to the bottom of the shared frame rather
    /// than sizing the window to itself, so that band has to fit inside the
    /// frame — otherwise `hud_chrome`'s fixed height would overflow the
    /// window and clip.
    #[test]
    fn hud_chrome_fits_within_the_shared_frame() {
        assert!(crate::overlay::HUD_CHROME_HEIGHT <= PANEL_HEIGHT);
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
                height: px(500.0),
            },
        };
        let origin = bottom_center_origin(display, panel_size(), PANEL_BOTTOM_MARGIN);
        // Unclamped this would be 500 - 480 - 120 = -100, off the top edge.
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
