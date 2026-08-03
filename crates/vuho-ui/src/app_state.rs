//! `UiCommand`: commands sent from non-GPUI contexts (the status-bar menu's
//! objc2 delegate, the provisioning thread) into the GPUI foreground task
//! that owns the panel window (`event_loop::spawn_ui_drain`).

use vuho_domain::ModelStatus;

/// Commands sent from non-GPUI contexts into the GPUI foreground task that
/// owns the panel.
pub(crate) enum UiCommand {
    /// A plain left click on the tray icon, or its menu's "Open Vuho" item.
    OpenPanel,
    /// A `vuho_model_fetch::ModelStatus` update from the provisioning
    /// thread — missing/downloading/verifying/failed model state. Drives
    /// both the status-bar toggle title (via `StatusModel`) and, for
    /// `Failed` while the panel isn't already open, surfaces the panel on
    /// the Settings tab.
    ModelStatus(ModelStatus),
    /// Engine warmup finished (only reachable once `ModelStatus::Ready` has
    /// been observed — see `wiring::load_engine_and_session`). `Ok` → the
    /// menu becomes usable; `Err` carries the failure to show, since a
    /// warmup failure means dictation can never start and the user must be
    /// told why.
    EngineReady(Result<(), String>),
}
