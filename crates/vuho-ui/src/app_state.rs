//! Process-lifetime GPUI global state for the production wiring.
//!
//! Holds everything the settings window and the status-bar menu need to
//! reach: the shared settings store, the (restartable) hotkey listener, the
//! command channel to the dictation session, and the settings window's
//! singleton handle.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use crossbeam_channel::Sender;
use gpui::{Entity, Global, WindowHandle};
use vuho_domain::{DictationCommand, ModelStatus};
use vuho_os_integration::HotkeyListener;
use vuho_settings::SettingsStore;

use crate::app_status::StatusModel;
use crate::settings_window::SettingsView;
use crate::wiring::ProvisionCommand;

/// Commands sent from non-GPUI contexts (the status-bar menu's objc2
/// delegate, the provisioning thread) into the GPUI foreground task that
/// owns window creation and the status-bar item.
pub(crate) enum UiCommand {
    /// Open (or re-activate) the settings window. No longer sent by the
    /// status-bar menu as of the WP4 click-split (see [`UiCommand::OpenPanel`])
    /// — kept, along with its `spawn_ui_drain` handler, for whatever
    /// reaches it next (the later unified-panel package, most likely).
    #[allow(
        dead_code,
        reason = "not currently sent — see the variant's doc comment"
    )]
    OpenSettings,
    /// Open (or re-activate) the readiness window (ADR-020). Same story as
    /// [`UiCommand::OpenSettings`]: the status-bar menu's "Setup…" item that
    /// used to send this was removed by the WP4 click-split.
    #[allow(
        dead_code,
        reason = "not currently sent — see the variant's doc comment"
    )]
    OpenReadiness,
    /// A plain left click on the tray icon, or its menu's "Open Vuho" item
    /// (WP4 click-split — see `status_bar.rs`'s module doc comment).
    /// Reaches the GPUI foreground task since the status-bar delegate has no
    /// `App` access of its own, same as `OpenSettings`/`OpenReadiness`.
    // TODO(ui-rehaul): route to the unified panel once it exists — for now
    // `spawn_ui_drain` opens the settings window, the closest existing
    // approximation.
    OpenPanel,
    /// A `vuho_model_fetch::ModelStatus` update from the provisioning
    /// thread — missing/downloading/verifying/failed model state. Drives
    /// both the status-bar toggle title and the readiness window's model
    /// row (`readiness::handle_model_status`).
    ModelStatus(ModelStatus),
    /// Engine warmup finished (only reachable once `ModelStatus::Ready` has
    /// been observed — see `wiring::load_engine_and_session`). `Ok` → the
    /// menu becomes usable; `Err` carries the failure to show, since a
    /// warmup failure means dictation can never start and the user must be
    /// told why.
    EngineReady(Result<(), String>),
}

/// Process-lifetime state shared between `main`'s production wiring, the
/// settings window, and the status-bar menu.
///
/// Registered once via `cx.set_global` — GPUI globals only require
/// `'static` (see [`gpui::Global`]), so the non-`Send` `Rc<RefCell<_>>`
/// here is legal as long as every access happens on the main thread, which
/// is true for all of this crate's production code.
pub(crate) struct VuhoState {
    /// The single settings-file owner for the process (CONSTITUTION rule 1).
    pub settings: Arc<SettingsStore>,
    /// Restartable: the settings window's hotkey-preset change live-rebinds
    /// this via `stop()` + `start()`.
    pub hotkey: Rc<RefCell<HotkeyListener>>,
    /// Shared with the hotkey listener and the status-bar menu — both
    /// funnel `DictationCommand`s into the same session.
    pub cmd_tx: Sender<DictationCommand>,
    /// The readiness window's Download/Retry button's only way to reach the
    /// provisioning thread (CONSTITUTION rule 20 — own both ends; the
    /// matching `Receiver` moved into `wiring::spawn_warmup_and_bridge`'s
    /// thread when this was constructed).
    pub provision_tx: Sender<ProvisionCommand>,
    /// Singleton settings window handle. `None` until first opened;
    /// [`crate::settings_window::open_settings_window`] resets this back to
    /// `None` via an `on_window_should_close` hook when the user closes the
    /// window, so it never holds a stale handle to a dead window between
    /// opens. `open_settings_window` also tolerates a stale handle in the
    /// unlikely case a window is torn down some other way: `handle.update`
    /// on a dead handle returns `Err`, which it treats as "not currently
    /// open" and falls through to opening (and storing) a fresh one.
    pub settings_window: Option<WindowHandle<SettingsView>>,
    /// The single source of truth for app state (UI rehaul) — settings
    /// window and status-bar wiring both write to this; `wiring::wire_production`
    /// registers the observer that keeps `status_bar::sync` reflecting it.
    pub status: Entity<StatusModel>,
}

impl Global for VuhoState {}
