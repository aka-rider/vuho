//! `UiCommand`: commands sent from non-GPUI contexts (the status-bar menu's
//! objc2 delegate, the provisioning thread) into the GPUI foreground task
//! that owns the panel window (`event_loop::spawn_ui_drain`).

use vuho_domain::ModelStatus;
use vuho_model_fetch::ModelAvailability;

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
    /// Every model the manifest knows, with the readiness the provisioning
    /// thread last observed for each — what the Settings tab's model list
    /// renders. Sent from the same single site as
    /// [`UiCommand::ModelStatus`], immediately after every phase
    /// transition, so the list and the selected model's status can never
    /// describe two different moments (see `wiring::Phase`'s doc comment).
    ModelList(Vec<ModelAvailability>),
    /// Engine warmup finished (only reachable once `ModelStatus::Ready` has
    /// been observed — see `wiring::load_engine_and_session`). `Ok` → the
    /// menu becomes usable; `Err` carries the failure to show, since a
    /// warmup failure means dictation can never start and the user must be
    /// told why.
    EngineReady(Result<(), String>),
}
