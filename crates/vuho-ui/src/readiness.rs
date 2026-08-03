//! Readiness window: the ADR-016 startup permission preflight, generalized
//! (ADR-020) to also surface Speech model download/verify progress.
//!
//! Two entry paths land in the same window implementation
//! (CONSTITUTION rule 26), and they stay deliberately distinct:
//!
//! - **Permissions unmet** — unchanged ADR-016 behavior. `main()` checks
//!   every required TCC grant *before* any real work (model warmup, hotkey
//!   start); if anything is missing, [`open_permission_gate_window`] shows
//!   one row per missing permission and, once everything is granted, a
//!   single "Relaunch Vuho" button — a TCC grant is a process-identity
//!   fact, so a fresh process is required (see
//!   `vuho_os_integration::prompt_accessibility_trust`'s doc comment). The
//!   `SpeechModel` requirement is **never** shown on this path: it runs
//!   before `wire_production`, so `VuhoState` — and the `ProvisionCommand`
//!   sender the Download button needs — doesn't exist yet.
//! - **Permissions granted, model missing** — does *not* block startup.
//!   Normal startup proceeds; [`handle_model_status`] (driven by
//!   `UiCommand::ModelStatus`) opens this same window showing the model row
//!   plus any permission later revoked mid-session (A5 — a permission can't
//!   be assumed to stay granted just because it was at startup). No
//!   relaunch on this path unless a permission needs one: a filesystem fact
//!   needs no new process, just a click.
//!
//! [`ReadinessView`] holds two independently-owned fields (CONSTITUTION
//! rule 1): `missing_permissions`, written only by [`spawn_poll_loop`], and
//! `model`, written only by [`handle_model_status`]. Reusing the original
//! `spawn_poll_loop`'s wholesale `view.missing = missing_permissions()`
//! assignment against a single combined field would have overwritten an
//! in-flight `Downloading 43%` row twice a second — this split is why that
//! can't happen.

use std::cell::RefCell;
use std::time::Duration;

use gpui::{
    div, hsla, prelude::*, px, App, Context, IntoElement, ParentElement, Pixels, Render,
    SharedString, Size, Styled, TitlebarOptions, Window, WindowBackgroundAppearance, WindowBounds,
    WindowHandle, WindowKind, WindowOptions,
};
use vuho_domain::ModelStatus;
use vuho_os_integration::InputMonitoringAccess;
use vuho_stt_engine::MicAuthStatus;

use crate::app_state::VuhoState;
use crate::event_loop::drain_pending;
use crate::permissions::{
    self, ACCESSIBILITY_SETTINGS_URL, INPUT_MONITORING_SETTINGS_URL, MICROPHONE_SETTINGS_URL,
};
use crate::wiring::ProvisionCommand;

/// Poll interval for re-checking permissions while the gate-mode window is
/// open. Matches `main.rs`'s `DRAIN_POLL_INTERVAL` order of magnitude —
/// frequent enough to feel live, cheap enough to not matter (three
/// synchronous TCC reads, no I/O).
const GATE_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Readiness window dimensions.
const GATE_WIDTH: Pixels = px(440.0);
const GATE_HEIGHT: Pixels = px(360.0);

/// Bytes per megabyte for the model-size display ("474 MB") — decimal MB,
/// matching the figure used throughout `ARCHITECTURE.md`/`CLAUDE.md`, not a
/// binary MiB.
const BYTES_PER_MB: u64 = 1_000_000;

// ── Access (tri-state OS grant status — CONSTITUTION rule 2: model it as
//    data, don't infer "denied" from click-then-poll timing) ──────────────

/// The tri-state status of one permission, as the OS actually reports it —
/// not collapsed to a bool. `Promptable` means "never asked, the native
/// one-click prompt will work"; `Denied` means the user (or an MDM policy)
/// already answered no, so re-firing the same prompt is a silent no-op and
/// the only way forward is System Settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    Granted,
    Promptable,
    Denied,
}

/// Map microphone `MicAuthStatus` to `Access`. Pure — unit-tested without
/// any macOS/TCC state (see `tests` below).
fn mic_access(status: MicAuthStatus) -> Access {
    match status {
        MicAuthStatus::Authorized => Access::Granted,
        MicAuthStatus::NotDetermined => Access::Promptable,
        MicAuthStatus::Denied | MicAuthStatus::Restricted => Access::Denied,
    }
}

/// Map Input Monitoring's `IOHIDCheckAccess` tri-state to `Access`. Pure.
fn input_monitoring_to_access(access: InputMonitoringAccess) -> Access {
    match access {
        InputMonitoringAccess::Granted => Access::Granted,
        InputMonitoringAccess::Unknown => Access::Promptable,
        InputMonitoringAccess::Denied => Access::Denied,
    }
}

/// Map Accessibility's `AXIsProcessTrusted` bool to `Access`.
///
/// **OS limitation, not a shortcut:** unlike Microphone (`AVFoundation`) and
/// Input Monitoring (`IOHIDCheckAccess`), the Accessibility API has no
/// three-state "not yet asked" vs "explicitly denied" distinction —
/// `AXIsProcessTrusted` is a plain bool. So this can only ever report
/// `Granted` or `Promptable`, never `Denied`. The gate's "Allow
/// Accessibility" button therefore always re-fires
/// `AXIsProcessTrustedWithOptions`, whose *own* native dialog (when the
/// grant was previously denied) already includes an "Open System
/// Settings…" button — the OS itself, not this app, handles that case for
/// Accessibility.
fn accessibility_access(trusted: bool) -> Access {
    if trusted {
        Access::Granted
    } else {
        Access::Promptable
    }
}

// ── Permission (the one data-driven definition — CONSTITUTION rule 26) ────

/// One of the three TCC grants Vuho needs, documented in `README.md`/
/// `CLAUDE.md`'s testing notes (`tccutil reset Microphone|Accessibility|InputMonitoring`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Permission {
    Accessibility,
    InputMonitoring,
    Microphone,
}

impl Permission {
    /// Every permission the gate checks, in the order shown to the user.
    pub(crate) const ALL: [Permission; 3] = [
        Permission::Accessibility,
        Permission::InputMonitoring,
        Permission::Microphone,
    ];

    /// Pure (non-prompting) tri-state check — safe to call on every poll tick.
    pub(crate) fn access(self) -> Access {
        match self {
            Permission::Accessibility => {
                accessibility_access(vuho_os_integration::accessibility_trusted())
            }
            Permission::InputMonitoring => {
                input_monitoring_to_access(vuho_os_integration::input_monitoring_access())
            }
            Permission::Microphone => mic_access(vuho_stt_engine::mic_permission_status()),
        }
    }

    /// This permission's System Settings deep-link, for the `Access::Denied`
    /// "Open System Settings" button. One source of truth for every URL —
    /// `permissions.rs` (CONSTITUTION rule 26); `show_microphone_denied`
    /// reuses the same `MICROPHONE_SETTINGS_URL` constant.
    pub(crate) fn settings_url(self) -> &'static str {
        match self {
            Permission::Accessibility => ACCESSIBILITY_SETTINGS_URL,
            Permission::InputMonitoring => INPUT_MONITORING_SETTINGS_URL,
            Permission::Microphone => MICROPHONE_SETTINGS_URL,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Permission::Accessibility => "Accessibility",
            Permission::InputMonitoring => "Input Monitoring",
            Permission::Microphone => "Microphone",
        }
    }

    pub(crate) fn description(self) -> &'static str {
        match self {
            Permission::Accessibility => {
                "Lets Vuho listen for the global CapsLock dictation hotkey."
            }
            Permission::InputMonitoring => {
                "Also required by the hotkey listener to receive keyboard events."
            }
            Permission::Microphone => "Lets Vuho capture your voice to transcribe.",
        }
    }

    /// Trigger this permission's native prompt. Fire-and-forget for all
    /// three: none of the underlying calls wait for the user's answer, so
    /// the gate's poll loop is what actually observes the grant landing.
    pub(crate) fn request(self) {
        match self {
            Permission::Accessibility => {
                let _ = vuho_os_integration::prompt_accessibility_trust();
            }
            Permission::InputMonitoring => vuho_os_integration::request_input_monitoring_access(),
            Permission::Microphone => {
                let _ = vuho_stt_engine::request_mic_permission();
            }
        }
    }
}

/// The preflight check: every currently-missing permission, in
/// [`Permission::ALL`] order. Side-effect-free — safe to call before any
/// other startup work, and repeatedly from the gate window's poll loop.
#[must_use]
pub(crate) fn missing_permissions() -> Vec<Permission> {
    Permission::ALL
        .into_iter()
        .filter(|p| p.access() != Access::Granted)
        .collect()
}

// ── Requirement (generalizes Permission to also cover the model —
//    CONSTITUTION rule 26: one row-building path for both entry modes) ────

/// One thing the readiness window can be waiting on.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Requirement {
    Permission(Permission),
    SpeechModel(ModelStatus),
}

/// Build the requirement rows a [`ReadinessView`] should render: every
/// currently-missing permission, plus a `SpeechModel` row when `model` is
/// present and not yet [`ModelStatus::Ready`] (a `Ready` model is nothing
/// left to require). Pure and free-standing so it's testable without GPUI
/// (see `tests` below) — [`ReadinessView::requirements`] is a thin wrapper
/// around this over `&self`'s fields.
fn combine_requirements(missing: &[Permission], model: Option<&ModelStatus>) -> Vec<Requirement> {
    let mut requirements: Vec<Requirement> = missing
        .iter()
        .copied()
        .map(Requirement::Permission)
        .collect();
    if let Some(status) = model {
        if *status != ModelStatus::Ready {
            requirements.push(Requirement::SpeechModel(status.clone()));
        }
    }
    requirements
}

// ── Readiness window ────────────────────────────────────────────────────

thread_local! {
    /// Main-thread-only singleton handle to the currently-open readiness
    /// window, so it can be recalled — or recreated, if the user closed it
    /// — from the status-bar menu ("Permissions…" in gate mode, "Setup…" in
    /// production mode). Mirrors `VuhoState::settings_window`'s singleton
    /// pattern (`settings_window.rs`), but the gate-mode entry runs
    /// *before* `VuhoState` exists (that global is set up by
    /// `wire_production`, which only ever runs once every permission is
    /// already granted), so this needs its own storage rather than reusing
    /// it.
    static READINESS_WINDOW: RefCell<Option<WindowHandle<ReadinessView>>> =
        const { RefCell::new(None) };

    /// Last `ModelStatus` observed, kept even while no window is open, so
    /// [`reopen_or_front_production_window`] (the "Setup…" menu item's
    /// target) can rebuild the model row immediately instead of waiting for
    /// the next progress tick. `None` for the whole process lifetime on the
    /// permission-gate path, and on the model path until the first
    /// `UiCommand::ModelStatus` arrives.
    static LAST_MODEL_STATUS: RefCell<Option<ModelStatus>> = const { RefCell::new(None) };

    /// Whether the user explicitly closed the readiness window (via its
    /// close button — [`open_readiness_window`]'s `on_window_should_close`
    /// hook, which only fires for user-initiated closes) while a
    /// requirement was still outstanding (A2). Reset to `false` every time
    /// a window is freshly opened; read by [`handle_model_status`] so a
    /// routine progress tick arriving after a dismissal records
    /// [`LAST_MODEL_STATUS`] without reopening the window and stealing
    /// focus back — a `Failed` status is the one exception, since that's
    /// the case the user actually needs to act on (Retry).
    static USER_DISMISSED: RefCell<bool> = const { RefCell::new(false) };
}

/// Which of the two entry paths a [`ReadinessView`] was opened from —
/// threaded through so [`spawn_poll_loop`] can tell "all requirements
/// satisfied" apart in a way that means two different things depending on
/// the path (A4/A5): gate mode shows "Relaunch Vuho" and stays open (a TCC
/// grant needs a fresh process to take effect); production mode has
/// nothing left to say and closes itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadinessMode {
    Gate,
    Production,
}

/// Command sent from the status-bar delegate's `showPermissions:` action
/// (an `AppKit` callback with no GPUI `App` access of its own) into
/// [`spawn_gate_command_drain`], the GPUI foreground drain that owns window
/// creation — mirrors `app_state::UiCommand`'s role for the settings window,
/// scoped to gate mode's one action.
pub(crate) enum GateCommand {
    /// Bring the readiness window to front, reopening it first if the user
    /// closed it.
    ReopenOrFront,
}

/// The readiness window's root view: one row per currently-missing
/// permission and/or the model's current status, or a single "Relaunch
/// Vuho" button once nothing is left to require.
///
/// `missing_permissions` and `model` are independently owned (CONSTITUTION
/// rule 1) — see this module's doc comment.
struct ReadinessView {
    missing_permissions: Vec<Permission>,
    model: Option<ModelStatus>,
    mode: ReadinessMode,
}

impl ReadinessView {
    fn requirements(&self) -> Vec<Requirement> {
        combine_requirements(&self.missing_permissions, self.model.as_ref())
    }
}

impl Render for ReadinessView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let requirements = self.requirements();
        let mut column = div()
            .flex()
            .flex_col()
            .size_full()
            .bg(hsla(0.7, 0.1, 0.14, 1.0))
            .text_color(hsla(0.0, 0.0, 1.0, 0.95))
            .p_6()
            .gap_5()
            .child(header_text());

        if requirements.is_empty() {
            column = column.child(render_relaunch_row(cx));
        } else {
            for requirement in requirements {
                column = column.child(render_requirement_row(requirement, cx));
            }
        }
        column
    }
}

/// Open the gate-mode readiness window listing `missing` (possibly empty —
/// see [`reopen_or_front_gate_window`], A4), and start its permission poll
/// loop. The `SpeechModel` row never appears on this path (`model: None`,
/// and nothing on this path ever calls [`handle_model_status`] — see this
/// module's doc comment).
pub(crate) fn open_permission_gate_window(missing: Vec<Permission>, cx: &mut App) {
    open_readiness_window(
        ReadinessView {
            missing_permissions: missing,
            model: None,
            mode: ReadinessMode::Gate,
        },
        cx,
    );
}

/// Open (or update, if already open) the readiness window for the
/// production-mode entry path — [`handle_model_status`] and
/// [`reopen_or_front_production_window`] — showing whichever of `missing`
/// and `model` currently apply (A5: a permission revoked mid-download must
/// show up here alongside the model row, not just the model row alone).
fn open_production_window(missing: Vec<Permission>, model: Option<ModelStatus>, cx: &mut App) {
    open_readiness_window(
        ReadinessView {
            missing_permissions: missing,
            model,
            mode: ReadinessMode::Production,
        },
        cx,
    );
}

/// Shared window-construction path for both entry modes (CONSTITUTION rule
/// 26) — modeled on `settings_window::open_settings_window`: centered,
/// fixed-size, non-resizable, `WindowKind::Normal`, with an
/// `on_window_should_close` hook that resets the [`READINESS_WINDOW`]
/// singleton back to `None` and records [`USER_DISMISSED`] (A2) at close
/// time, so a later reopen never finds a stale handle to a dead window and
/// [`handle_model_status`] knows the close was the user's doing. Both modes
/// get a live permission poll loop now (A5 — production windows used to
/// have none at all).
fn open_readiness_window(view: ReadinessView, cx: &mut App) {
    let mode = view.mode;
    let size = Size {
        width: GATE_WIDTH,
        height: GATE_HEIGHT,
    };
    let bounds = WindowBounds::centered(size, cx);
    let result = cx.open_window(
        WindowOptions {
            window_bounds: Some(bounds),
            titlebar: Some(TitlebarOptions {
                title: Some(SharedString::from("Vuho setup")),
                appears_transparent: false,
                traffic_light_position: None,
            }),
            focus: true,
            show: true,
            kind: WindowKind::Normal,
            is_resizable: false,
            is_minimizable: false,
            window_background: WindowBackgroundAppearance::Opaque,
            ..Default::default()
        },
        |window, cx| {
            window.set_window_title("Vuho setup");
            window.on_window_should_close(cx, |_window, _cx| {
                READINESS_WINDOW.with(|g| *g.borrow_mut() = None);
                USER_DISMISSED.with(|d| *d.borrow_mut() = true);
                true
            });
            cx.new(|_cx| view)
        },
    );

    match result {
        Ok(handle) => {
            READINESS_WINDOW.with(|g| *g.borrow_mut() = Some(handle));
            USER_DISMISSED.with(|d| *d.borrow_mut() = false);
            front_readiness_window();
            spawn_poll_loop(handle, mode, cx);
        }
        Err(e) => log::error!("readiness: failed to open window: {e}"),
    }
}

/// If the readiness window is alive, bring it to front and report so —
/// shared by both `reopen_or_front_*` entry points below (CONSTITUTION rule
/// 26).
fn front_if_alive(cx: &mut App) -> bool {
    let alive = READINESS_WINDOW
        .with(|g| *g.borrow())
        .is_some_and(|handle| {
            handle
                .update(cx, |_, window, _| window.activate_window())
                .is_ok()
        });
    if alive {
        front_readiness_window();
    }
    alive
}

/// The "Permissions…" menu item's target (gate mode): reopen (or re-front)
/// the gate window. Unlike the production path, this **always** opens
/// something when asked — even "All permissions granted, Relaunch Vuho"
/// (A4) when nothing is currently missing — since the user explicitly
/// clicked to see it, and gate mode never auto-closes itself (a TCC grant
/// needs a fresh process to take effect, so there is always a next step to
/// show).
pub(crate) fn reopen_or_front_gate_window(cx: &mut App) {
    if front_if_alive(cx) {
        return;
    }
    open_permission_gate_window(missing_permissions(), cx);
}

/// The "Setup…" menu item's target (production mode): reopen (or re-front)
/// the production window for whatever's currently missing — permission(s),
/// the model, or both. Fixes A5: this used to only ever consult
/// [`LAST_MODEL_STATUS`] and build a model-only window with an empty
/// permission list and no poll loop, so a permission revoked mid-download
/// never surfaced here at all.
pub(crate) fn reopen_or_front_production_window(cx: &mut App) {
    if front_if_alive(cx) {
        return;
    }

    let missing = missing_permissions();
    let model = LAST_MODEL_STATUS
        .with(|s| s.borrow().clone())
        .filter(|status| *status != ModelStatus::Ready);
    if missing.is_empty() && model.is_none() {
        // Model already `Ready` (or no status observed yet) and no
        // permission missing: genuinely nothing to show — not the
        // gate-mode "everything granted, offer Relaunch" state, which only
        // applies to the permission-gate path.
        log::info!("readiness: nothing to show — every requirement already satisfied");
        return;
    }
    open_production_window(missing, model, cx);
}

/// React to a `ModelStatus` update reaching the GPUI foreground task
/// (`event_loop::spawn_ui_drain`'s `UiCommand::ModelStatus` arm): keep
/// [`LAST_MODEL_STATUS`] current, then either update the open window's
/// `model` field, close it (`Ready` — nothing left to show), or open a
/// fresh one — so a download in progress is never silently invisible, even
/// if the user closed the window earlier: the next progress tick brings it
/// back. This is the **only** function that writes `ReadinessView::model`
/// (CONSTITUTION rule 1 — mirrors [`spawn_poll_loop`] being the only writer
/// of `missing_permissions`).
///
/// A2 fix: "brings it back" used to mean *every* status update reopened a
/// closed window and stole focus (`open_readiness_window` → `activate_app`)
/// — so closing the window mid-download reopened it within the next
/// progress tick, unconditionally. Now a dismissed window ([`USER_DISMISSED`])
/// only reopens for a status the user must actually act on ([`ModelStatus::Failed`]);
/// a routine `Downloading`/`Verifying` tick just updates [`LAST_MODEL_STATUS`]
/// silently, the same as if no window had ever been opened at all.
pub(crate) fn handle_model_status(status: ModelStatus, cx: &mut App) {
    LAST_MODEL_STATUS.with(|s| *s.borrow_mut() = Some(status.clone()));

    if status == ModelStatus::Ready {
        close_window_if_open(cx);
        return;
    }

    let updated = READINESS_WINDOW
        .with(|g| *g.borrow())
        .is_some_and(|handle| {
            handle
                .update(cx, |view, _window, cx| {
                    view.model = Some(status.clone());
                    cx.notify();
                })
                .is_ok()
        });
    if updated {
        return;
    }

    let dismissed = USER_DISMISSED.with(|d| *d.borrow());
    if dismissed && !matches!(status, ModelStatus::Failed { .. }) {
        log::info!("readiness: window was dismissed by the user — not reopening for {status:?}");
        return;
    }

    open_production_window(missing_permissions(), Some(status), cx);
}

/// Close the readiness window if one is open, clearing the singleton first
/// — `Window::remove_window` bypasses `on_window_should_close` (that hook
/// only fires for user-initiated closes), so without clearing it here the
/// singleton would keep pointing at a window that's about to stop existing.
fn close_window_if_open(cx: &mut App) {
    let handle = READINESS_WINDOW.with(|g| g.borrow_mut().take());
    if let Some(handle) = handle {
        let _ = handle.update(cx, |_view, window, _cx| window.remove_window());
    }
}

/// Drain [`GateCommand`]s sent from the status-bar delegate's
/// `showPermissions:` action into [`reopen_or_front_gate_window`] calls.
///
/// Mirrors `event_loop::spawn_ui_drain`'s poll-and-detach shape (production
/// mode's equivalent `AppKit`-callback-to-GPUI bridge) — gate mode can't
/// reuse that drain directly since it's typed to `app_state::UiCommand`, and
/// gate mode runs before `wire_production` (which owns that channel) exists.
pub(crate) fn spawn_gate_command_drain(
    gate_rx: crossbeam_channel::Receiver<GateCommand>,
    cx: &mut App,
) {
    cx.spawn(move |cx: &mut gpui::AsyncApp| {
        let cx = cx.clone();
        async move {
            loop {
                let Some(commands) = drain_pending(&gate_rx) else {
                    log::info!("readiness: gate command channel disconnected — stopping drain");
                    return;
                };
                if !commands.is_empty() {
                    let updated = cx.update(reopen_or_front_gate_window);
                    if updated.is_err() {
                        log::info!("readiness: app context gone — stopping gate command drain");
                        return;
                    }
                }
                cx.background_executor().timer(GATE_POLL_INTERVAL).await;
            }
        }
    })
    .detach();
}

/// Bring the accessory app forward so the just-opened readiness window
/// (`focus: true`) actually orders in front of whatever else is on screen.
///
/// The app runs under `NSApplicationActivationPolicyAccessory` (set before
/// this is ever called — see `main.rs`), which does not activate on its own
/// just because a window requests focus. Without this, the window is a
/// real, focused window that nonetheless renders *behind* the frontmost
/// app — effectively invisible on a fresh install.
///
/// Also called from the menu-bar "Permissions…"/"Setup…" items so the
/// window can always be recalled to front.
fn front_readiness_window() {
    let Some(mtm) = objc2::MainThreadMarker::new() else {
        return;
    };
    crate::permissions::activate_app(mtm);
}

/// Re-check `missing_permissions()` every [`GATE_POLL_INTERVAL`] and write
/// the result into the view's `missing_permissions` field only (never
/// `model` — CONSTITUTION rule 1), so granted permissions disappear from
/// the list live, in **both** modes now (A5 — production windows used to
/// get no poll loop at all).
///
/// Mirrors `main.rs::spawn_event_drain`'s poll-and-detach shape. Stops (does
/// not keep ticking for the rest of the process lifetime) once `handle.update`
/// starts returning `Err` — the user closed the window without granting
/// everything (CONSTITUTION rule 10) — or, in [`ReadinessMode::Production`]
/// only, once every requirement is satisfied: that mode has nothing left to
/// say once its row list is empty, so it closes itself rather than sitting
/// on screen with a "Relaunch Vuho" button that means nothing outside gate
/// mode (A4/A5). Gate mode never auto-closes: an empty row list there means
/// "Relaunch Vuho", not "nothing to show".
fn spawn_poll_loop(handle: WindowHandle<ReadinessView>, mode: ReadinessMode, cx: &mut App) {
    cx.spawn(move |cx: &mut gpui::AsyncApp| {
        let mut cx = cx.clone();
        async move {
            loop {
                cx.background_executor().timer(GATE_POLL_INTERVAL).await;
                let missing = missing_permissions();
                let empty_now = handle.update(&mut cx, |view, _window, cx| {
                    view.missing_permissions = missing;
                    cx.notify();
                    view.requirements().is_empty()
                });
                match empty_now {
                    Ok(true) if mode == ReadinessMode::Production => {
                        let closed = cx.update(close_window_if_open);
                        if closed.is_err() {
                            log::info!(
                                "readiness: app context gone while auto-closing — stopping poll"
                            );
                        } else {
                            log::info!(
                                "readiness: production window auto-closed — nothing left to require"
                            );
                        }
                        return;
                    }
                    Ok(_) => {}
                    Err(_) => {
                        log::info!("readiness: window closed — stopping permission poll");
                        return;
                    }
                }
            }
        }
    })
    .detach();
}

// ── Rendering helpers (CONSTITUTION rule 28: split, ≤40 lines each) ───────

fn header_text() -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .text_size(px(16.0))
                .child("Vuho needs a couple of things"),
        )
        .child(
            div()
                .text_size(px(12.0))
                .text_color(hsla(0.0, 0.0, 1.0, 0.6))
                .child("Handle each item below."),
        )
}

/// One requirement row, dispatched to the permission or model renderer.
fn render_requirement_row(
    requirement: Requirement,
    cx: &mut Context<ReadinessView>,
) -> gpui::AnyElement {
    match requirement {
        Requirement::Permission(permission) => {
            render_permission_row(permission, cx).into_any_element()
        }
        Requirement::SpeechModel(status) => render_speech_model_row(&status, cx),
    }
}

/// One permission row: label, description, and an action button whose
/// label/behavior depends on this permission's current `Access` — a native
/// "Allow …" prompt when `Promptable`, an "Open System Settings" deep-link
/// when `Denied` (re-firing a denied prompt is a silent no-op, see `Access`'s
/// doc comment).
fn render_permission_row(
    permission: Permission,
    cx: &mut Context<ReadinessView>,
) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .p_3()
        .rounded(px(6.0))
        .bg(hsla(0.0, 0.0, 1.0, 0.06))
        .child(div().text_size(px(14.0)).child(permission.label()))
        .child(
            div()
                .text_size(px(12.0))
                .text_color(hsla(0.0, 0.0, 1.0, 0.6))
                .child(permission.description()),
        )
        .child(action_button(permission, cx))
}

/// One speech-model row: label, a status subtitle (size before download,
/// live percentage while downloading, or the failure message), and a
/// Download/Retry button — or disabled progress text while a download is
/// in flight.
fn render_speech_model_row(
    status: &ModelStatus,
    cx: &mut Context<ReadinessView>,
) -> gpui::AnyElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .p_3()
        .rounded(px(6.0))
        .bg(hsla(0.0, 0.0, 1.0, 0.06))
        .child(div().text_size(px(14.0)).child("Speech model"))
        .child(
            div()
                .text_size(px(12.0))
                .text_color(hsla(0.0, 0.0, 1.0, 0.6))
                .child(model_status_text(status)),
        )
        .child(model_action_button(status, cx))
        .into_any_element()
}

/// Human-readable subtitle for a model-status row.
pub(crate) fn model_status_text(status: &ModelStatus) -> String {
    match status {
        ModelStatus::Missing { total_bytes } => {
            format!("{} · not yet downloaded", format_mb(*total_bytes))
        }
        ModelStatus::Downloading {
            received_bytes,
            total_bytes,
        } => format!(
            "Downloading… {} of {}",
            format_mb(*received_bytes),
            format_mb(*total_bytes)
        ),
        ModelStatus::Verifying => "Verifying…".to_owned(),
        // Unreachable in practice — `combine_requirements` filters `Ready`
        // out before a row is ever built for it.
        ModelStatus::Ready => "Ready".to_owned(),
        ModelStatus::Failed { message } => message.clone(),
    }
}

/// Format a byte count as a whole number of megabytes (decimal, "474 MB") —
/// readable, unlike the raw byte count `models.lock.json` stores.
pub(crate) fn format_mb(bytes: u64) -> String {
    format!("{} MB", (bytes + BYTES_PER_MB / 2) / BYTES_PER_MB)
}

/// The model row's action: `Download` (`Missing`), `Retry` (`Failed`, with
/// the failure message already shown by [`model_status_text`]), or disabled
/// progress text (`Downloading`/`Verifying`).
fn model_action_button(status: &ModelStatus, cx: &mut Context<ReadinessView>) -> gpui::AnyElement {
    match status {
        ModelStatus::Missing { .. } => model_download_button("Download", cx),
        ModelStatus::Failed { .. } => model_download_button("Retry", cx),
        ModelStatus::Downloading { .. } | ModelStatus::Verifying => {
            button_base(hsla(0.0, 0.0, 1.0, 0.08))
                .id("readiness-model-progress")
                .mt_1()
                .child("In progress…")
                .into_any_element()
        }
        // Unreachable — see `model_status_text`'s `Ready` arm.
        ModelStatus::Ready => div().into_any_element(),
    }
}

/// The clickable Download/Retry button: sends [`ProvisionCommand::Download`]
/// on `VuhoState`'s sender (only reachable here — this row only ever
/// renders in production mode, after `wire_production` has registered the
/// global).
fn model_download_button(label: &'static str, cx: &mut Context<ReadinessView>) -> gpui::AnyElement {
    button_base(hsla(0.55, 0.5, 0.45, 1.0))
        .id("readiness-model-download")
        .mt_1()
        .child(label)
        .on_click(cx.listener(|_view, _event, _window, cx| {
            let provision_tx = cx.global::<VuhoState>().provision_tx.clone();
            let _ = provision_tx.send(ProvisionCommand::Download);
        }))
        .into_any_element()
}

/// Base style shared by the readiness window's clickable/disabled-looking
/// buttons — the one source of truth for their common look (CONSTITUTION
/// rule 26). Callers supply the background color and chain their own
/// id/label/click handler.
fn button_base(bg: gpui::Hsla) -> gpui::Div {
    div().px_3().py_2().rounded(px(6.0)).bg(bg).cursor_pointer()
}

/// The action button for one permission row — driven by this permission's
/// current `Access`: a native "Allow …" prompt when `Promptable`, an "Open
/// System Settings" deep-link when `Denied`. Either way the poll loop (not
/// this handler) observes the eventual grant; a settled `Granted`
/// permission never reaches this function (`missing_permissions` already
/// excludes it).
fn action_button(permission: Permission, cx: &mut Context<ReadinessView>) -> impl IntoElement {
    let button = button_base(hsla(0.55, 0.5, 0.45, 1.0))
        .id(("permission-gate-action", permission as usize))
        .mt_1();

    match permission.access() {
        Access::Denied => button.child("Open System Settings").on_click(cx.listener(
            move |_view, _event, _window, _cx| {
                permissions::open_url(permission.settings_url());
            },
        )),
        // Granted rows are filtered out of `missing_permissions` before
        // reaching this function; treat it the same as Promptable rather
        // than panic, so a race with the poll loop degrades gracefully.
        Access::Promptable | Access::Granted => button
            .child(format!("Allow {}", permission.label()))
            .on_click(cx.listener(move |_view, _event, _window, _cx| {
                permission.request();
            })),
    }
}

/// The "all requirements satisfied" state: a confirmation line and the
/// relaunch button. Only ever rendered on the gate-mode (permission) path —
/// see `reopen_or_front_production_window`'s doc comment for why a production-mode
/// window with nothing to show never opens in the first place, so this
/// never renders with "Relaunch" in a context where a relaunch wouldn't
/// mean anything.
fn render_relaunch_row(cx: &mut Context<ReadinessView>) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .child(div().child("All permissions granted."))
        .child(
            button_base(hsla(0.35, 0.5, 0.4, 1.0))
                .id("permission-gate-relaunch")
                .child("Relaunch Vuho")
                .on_click(cx.listener(|_view, _event, _window, _cx| relaunch())),
        )
}

/// Re-exec the current binary and exit this process.
///
/// Works identically for `cargo run`'s raw binary and the packaged `.app`'s
/// binary — `current_exe()` resolves to the actual executable path in both
/// cases, no bundle-path logic needed. Only exits if the spawn actually
/// succeeded, so a failed relaunch doesn't strand the user with no window at
/// all.
pub(crate) fn relaunch() {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            log::error!("readiness: could not resolve current_exe for relaunch: {e}");
            return;
        }
    };
    match std::process::Command::new(&exe).spawn() {
        Ok(_) => std::process::exit(0),
        Err(e) => log::error!("readiness: failed to relaunch {}: {e}", exe.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_all_covers_every_variant() {
        assert_eq!(Permission::ALL.len(), 3);
        assert!(Permission::ALL.contains(&Permission::Accessibility));
        assert!(Permission::ALL.contains(&Permission::InputMonitoring));
        assert!(Permission::ALL.contains(&Permission::Microphone));
    }

    #[test]
    fn permission_labels_are_distinct_and_nonempty() {
        let labels: Vec<&str> = Permission::ALL.iter().map(|p| p.label()).collect();
        for label in &labels {
            assert!(!label.is_empty());
        }
        let mut unique = labels.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            labels.len(),
            "labels must be distinct: {labels:?}"
        );
    }

    #[test]
    fn permission_descriptions_are_nonempty() {
        for permission in Permission::ALL {
            assert!(!permission.description().is_empty());
        }
    }

    /// Pure, non-prompting: must never panic even in a headless/CI
    /// environment with no TCC state at all.
    #[test]
    fn missing_permissions_does_not_panic() {
        let _ = missing_permissions();
    }

    // ── Pure Access mappings — headless-safe, no macOS/TCC state involved.

    #[test]
    fn mic_access_covers_every_source_variant() {
        assert_eq!(mic_access(MicAuthStatus::Authorized), Access::Granted);
        assert_eq!(mic_access(MicAuthStatus::NotDetermined), Access::Promptable);
        assert_eq!(mic_access(MicAuthStatus::Denied), Access::Denied);
        assert_eq!(mic_access(MicAuthStatus::Restricted), Access::Denied);
    }

    #[test]
    fn input_monitoring_to_access_covers_every_source_variant() {
        assert_eq!(
            input_monitoring_to_access(InputMonitoringAccess::Granted),
            Access::Granted
        );
        assert_eq!(
            input_monitoring_to_access(InputMonitoringAccess::Unknown),
            Access::Promptable
        );
        assert_eq!(
            input_monitoring_to_access(InputMonitoringAccess::Denied),
            Access::Denied
        );
    }

    #[test]
    fn accessibility_access_covers_every_source_variant() {
        assert_eq!(accessibility_access(true), Access::Granted);
        assert_eq!(accessibility_access(false), Access::Promptable);
    }

    /// Every permission's settings URL is a well-formed
    /// `x-apple.systempreferences:` deep-link, and distinct per permission.
    #[test]
    fn settings_urls_are_distinct_deep_links() {
        let urls: Vec<&str> = Permission::ALL.iter().map(|p| p.settings_url()).collect();
        for url in &urls {
            assert!(
                url.starts_with("x-apple.systempreferences:"),
                "not a deep-link: {url}"
            );
        }
        let mut unique = urls.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            urls.len(),
            "settings URLs must be distinct: {urls:?}"
        );
    }

    /// Pure, non-prompting: `Permission::access()` must never panic headless.
    #[test]
    fn permission_access_does_not_panic() {
        for permission in Permission::ALL {
            let _ = permission.access();
        }
    }

    // ── Requirement / combine_requirements — pure, no GPUI involved
    //    (the rule-1 regression: a model update must never clobber
    //    `missing_permissions`, and vice versa — enforced here structurally,
    //    since `combine_requirements` only ever reads both, never a shared
    //    mutable field).

    #[test]
    fn combine_requirements_includes_missing_permissions_and_a_non_ready_model() {
        let missing = vec![Permission::Accessibility, Permission::Microphone];
        let model = ModelStatus::Missing {
            total_bytes: 474_000_000,
        };
        let reqs = combine_requirements(&missing, Some(&model));
        assert_eq!(reqs.len(), 3);
        assert!(matches!(
            reqs[0],
            Requirement::Permission(Permission::Accessibility)
        ));
        assert!(matches!(
            reqs[1],
            Requirement::Permission(Permission::Microphone)
        ));
        assert!(matches!(reqs[2], Requirement::SpeechModel(_)));
    }

    #[test]
    fn combine_requirements_omits_a_ready_model() {
        let reqs = combine_requirements(&[], Some(&ModelStatus::Ready));
        assert!(reqs.is_empty());
    }

    #[test]
    fn combine_requirements_omits_model_row_when_none() {
        let missing = vec![Permission::Microphone];
        let reqs = combine_requirements(&missing, None);
        assert_eq!(reqs.len(), 1);
        assert!(matches!(
            reqs[0],
            Requirement::Permission(Permission::Microphone)
        ));
    }

    #[test]
    fn combine_requirements_with_nothing_missing_and_no_model_is_empty() {
        assert!(combine_requirements(&[], None).is_empty());
    }

    #[test]
    fn format_mb_rounds_to_the_nearest_megabyte() {
        assert_eq!(format_mb(474_000_000), "474 MB");
        assert_eq!(format_mb(496_210_831), "496 MB");
        assert_eq!(format_mb(0), "0 MB");
    }
}
