//! `StatusModel`: the single source of truth for app state, feeding the
//! tray icon/menu (`status_bar.rs`), the panel's idle status block
//! (`crate::panel::PanelRoot::render_idle_status`), and the Settings tab
//! (`crate::settings_tab::SettingsTab`).
//!
//! [`StatusModel::composite`]/[`CompositeStatus::menu_title`]/
//! [`CompositeStatus::toggle_enabled`] drive the tray;
//! [`StatusModel::idle_headline`] drives the panel's idle status block.
//!
//! `StatusModel` itself stays a plain struct — no `Global` impl, no
//! channels — so it composes as a GPUI `Entity` (`cx.new(|_| StatusModel {
//! .. })`), shared by reference (`Entity::clone`) with the Settings tab and
//! the panel root.

use gpui::SharedString;
use vuho_domain::ModelStatus;
use vuho_settings::HotkeySetting;

use crate::readiness::{self, Access, Permission};

/// STT engine warmup state (`wiring::load_engine_and_session`'s outcome),
/// mirrored here so `StatusModel` doesn't need a reference to the engine
/// itself — just what it reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EngineState {
    /// Warmup in progress; a dictation press does nothing yet.
    Loading,
    /// Warmup succeeded; dictation can start.
    Ready,
    /// Warmup failed; dictation can never start this run. Carries the
    /// failure message from `wiring`'s `UiCommand::EngineReady(Err(_))`.
    Failed(String),
}

/// Whether the configured global hotkey is actually listening.
///
/// A `HotkeyListener` can fail to (re)start — e.g. the settings window
/// live-rebinds it (`stop()` + `start()`) and the new chord's `CGEventTap`
/// setup fails — leaving dictation reachable only via the menu-bar toggle.
/// Carries the `HotkeySetting` in both arms so [`StatusModel::idle_headline`]
/// always has the preset to render, whether or not the tap is actually live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HotkeyState {
    Active(HotkeySetting),
    Failed(HotkeySetting),
}

/// The single source of truth for app state — feeds the tray icon/menu, the
/// panel's idle status block, and the Settings tab.
///
/// Plain struct: this is meant to be wrapped in a GPUI `Entity` by its
/// caller, not to own GPUI machinery itself.
pub(crate) struct StatusModel {
    /// `None` until the provisioning loop first reports a status (gate mode
    /// — the permission-gate startup path, which never reaches model
    /// provisioning — stays `None` for the whole process lifetime).
    pub model: Option<ModelStatus>,
    /// Initial value: [`EngineState::Loading`] — warmup starts immediately
    /// and unconditionally.
    pub engine: EngineState,
    pub recording: bool,
    pub hotkey: HotkeyState,
    /// Every permission still missing, each with its live [`Access`] —
    /// derived exclusively by `readiness::missing_permissions()` (F6).
    /// Three writer *sites*, one derivation (G7 added the second of them):
    /// `main.rs`'s `run_gate_blocked` one-time gate-path seed (writes
    /// `readiness::missing_permissions()` inline — it runs before the panel
    /// exists, so it can't call the shared helper below), and
    /// `crate::panel`'s `show_full` (a synchronous seed on every Full-
    /// presentation open, through the private `refresh_permissions_missing`
    /// helper) and `start_permissions_poll` (that same helper, on every
    /// tick) — the latter two are the same derivation by construction, so
    /// they can never disagree in shape with each other, and all three read
    /// the identical `readiness::missing_permissions()` call. The seeds
    /// exist only so a Settings-tab-showing first paint — the tray's, via
    /// `run_gate_blocked` (`main.rs`), or the panel's own, via `show_full`
    /// (G7) — never renders one visibly wrong frame from a field the async
    /// poll hasn't had a chance to write to yet (`launch_blocked` true with
    /// an empty `permissions_missing` derives
    /// [`CompositeStatus::RelaunchRequired`], not
    /// [`CompositeStatus::PermissionsMissing`]); every write after a seed is
    /// the poll's alone. `settings_tab.rs`'s permission rows render purely
    /// from this field (plus [`Permission::ALL`] for the granted rows) —
    /// never a fresh `Permission::access()` call at render time, which used
    /// to race the poll's own 500 ms tick.
    pub permissions_missing: Vec<(Permission, Access)>,
    /// `true` when the app started on the permission-gate path — a relaunch
    /// is required after granting for the new process identity to carry
    /// the TCC grant (see `readiness.rs`'s module doc comment).
    pub launch_blocked: bool,
    /// Surfaced as a warning banner at the top of the Settings tab
    /// (`crate::settings_tab::SettingsTab::render`) when the persisted
    /// settings file couldn't be read and defaults were substituted.
    pub settings_load_warning: Option<SharedString>,
}

/// The one composite state derived from every [`StatusModel`] field, in
/// strict priority order (highest first): a higher-priority condition being
/// true always wins, regardless of what a lower-priority field says —
/// e.g. a live session with a missing permission still reports
/// [`CompositeStatus::Recording`], not [`CompositeStatus::PermissionsMissing`]
/// (F8/F18): a live session's Stop control must stay reachable and visible
/// no matter what else is going on; a permission revoked mid-session (A5,
/// `readiness.rs`) surfaces once the session actually ends, not by yanking
/// the Stop control out from under a user who is still mid-dictation.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum CompositeStatus {
    /// A dictation session is in progress — outranks everything else (see
    /// this enum's doc comment).
    Recording,
    PermissionsMissing,
    /// `launch_blocked` and every permission is now granted — the process
    /// itself needs to be relaunched (a TCC grant is a process-identity
    /// fact) before startup can proceed past the gate.
    RelaunchRequired,
    /// [`ModelStatus::Missing`] or [`ModelStatus::Failed`] — both need the
    /// same next user action (open the readiness window, click
    /// Download/Retry), so they collapse to one composite state.
    ModelMissing,
    /// Rounded download percent, `0..=100`.
    Downloading(u8),
    Verifying,
    EngineFailed,
    EngineLoading,
    Ready,
}

impl StatusModel {
    /// Derive the one composite state per the priority order documented on
    /// [`CompositeStatus`].
    #[must_use]
    pub(crate) fn composite(&self) -> CompositeStatus {
        if self.recording {
            return CompositeStatus::Recording;
        }
        if !self.permissions_missing.is_empty() {
            return CompositeStatus::PermissionsMissing;
        }
        if self.launch_blocked {
            return CompositeStatus::RelaunchRequired;
        }
        if let Some(model_composite) = self.model.as_ref().and_then(model_composite_status) {
            return model_composite;
        }
        match &self.engine {
            EngineState::Failed(_) => return CompositeStatus::EngineFailed,
            EngineState::Loading => return CompositeStatus::EngineLoading,
            EngineState::Ready => {}
        }
        CompositeStatus::Ready
    }

    /// Headline + optional sub-line for the panel's idle status block, one
    /// per [`CompositeStatus`] — computed via [`StatusModel::composite`] so
    /// this never drifts from the priority order it defines.
    #[must_use]
    pub(crate) fn idle_headline(&self) -> (SharedString, Option<SharedString>) {
        match self.composite() {
            CompositeStatus::Recording => (SharedString::from("Listening…"), None),
            CompositeStatus::PermissionsMissing => (
                SharedString::from("Permissions needed"),
                Some(SharedString::from(
                    "Grant the permissions Vuho needs to dictate.",
                )),
            ),
            CompositeStatus::RelaunchRequired => (
                SharedString::from("Relaunch required"),
                Some(SharedString::from(
                    "Relaunch Vuho for the granted permissions to take effect.",
                )),
            ),
            CompositeStatus::ModelMissing => model_missing_headline(self.model.as_ref()),
            CompositeStatus::Downloading(pct) => (
                SharedString::from("Downloading speech model…"),
                Some(SharedString::from(format!("{pct}% complete"))),
            ),
            CompositeStatus::Verifying => {
                (SharedString::from("Verifying speech model…"), None)
            }
            CompositeStatus::EngineFailed => engine_failed_headline(&self.engine),
            // F7: distinct from the menu title ("Loading model…", unchanged
            // in `CompositeStatus::menu_title` below) — the matrix calls for
            // separate copy here, since "Loading model…" during ANE warmup
            // (no bytes moving, no percent to show) reads like a stalled
            // download rather than what it actually is.
            CompositeStatus::EngineLoading => (
                SharedString::from("Getting ready"),
                Some(SharedString::from("Warming up the speech engine…")),
            ),
            CompositeStatus::Ready => ready_headline(self.hotkey),
        }
    }
}

/// Map a non-`Ready` [`ModelStatus`] to its [`CompositeStatus`], or `None`
/// for [`ModelStatus::Ready`] (nothing model-related to report — the
/// engine/recording states below take over).
fn model_composite_status(status: &ModelStatus) -> Option<CompositeStatus> {
    match status {
        ModelStatus::Missing { .. } | ModelStatus::Failed { .. } => {
            Some(CompositeStatus::ModelMissing)
        }
        ModelStatus::Downloading { .. } => {
            let fraction = status.fraction().unwrap_or(0.0).clamp(0.0, 1.0);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "fraction is clamped to [0.0, 1.0], so *100.0 fits u8 exactly"
            )]
            let pct = (fraction * 100.0).round() as u8;
            Some(CompositeStatus::Downloading(pct))
        }
        ModelStatus::Verifying => Some(CompositeStatus::Verifying),
        ModelStatus::Ready => None,
    }
}

/// [`CompositeStatus::ModelMissing`]'s headline: mention the download size
/// via [`readiness::format_mb`] when the model is simply [`ModelStatus::Missing`]
/// (the size is known up front from the pinned lockfile — see that variant's
/// doc comment); show the failure message for [`ModelStatus::Failed`].
fn model_missing_headline(model: Option<&ModelStatus>) -> (SharedString, Option<SharedString>) {
    let sub = match model {
        Some(ModelStatus::Missing { total_bytes }) => Some(SharedString::from(format!(
            "Download the {} speech model to get started.",
            readiness::format_mb(*total_bytes)
        ))),
        Some(ModelStatus::Failed { message }) => Some(SharedString::from(message.clone())),
        _ => None,
    };
    (SharedString::from("Speech model needed"), sub)
}

/// [`CompositeStatus::EngineFailed`]'s headline: surface the warmup failure
/// message from [`EngineState::Failed`] as the sub-line.
fn engine_failed_headline(engine: &EngineState) -> (SharedString, Option<SharedString>) {
    let sub = match engine {
        EngineState::Failed(message) => Some(SharedString::from(message.clone())),
        EngineState::Loading | EngineState::Ready => None,
    };
    (SharedString::from("Engine unavailable"), sub)
}

/// [`CompositeStatus::Ready`]'s headline: an inactive-hotkey warning, or
/// "press `<label>` to dictate" with the label read from the existing
/// [`HotkeySetting::label`] source (`hotkey_presets.rs` renders the same
/// presets) — never hardcoded, since the preset is user-configurable.
fn ready_headline(hotkey: HotkeyState) -> (SharedString, Option<SharedString>) {
    match hotkey {
        HotkeyState::Failed(_) => (
            SharedString::from("Hotkey inactive — use the menu-bar icon, or fix in Settings"),
            None,
        ),
        HotkeyState::Active(preset) => (
            SharedString::from(format!("Press {} to dictate", preset.label())),
            None,
        ),
    }
}

impl CompositeStatus {
    /// The tray menu-item title for this composite state — subsumes
    /// `status_bar::AppStatus::title()`; wording reused verbatim where an
    /// equivalent `AppStatus` variant exists so the menu text doesn't churn
    /// once this replaces it.
    #[must_use]
    pub(crate) fn menu_title(&self) -> String {
        match self {
            CompositeStatus::PermissionsMissing => "Permissions…".to_owned(),
            CompositeStatus::RelaunchRequired => "Relaunch required".to_owned(),
            CompositeStatus::ModelMissing => "Model setup needed".to_owned(),
            CompositeStatus::Downloading(pct) => format!("Downloading model… {pct}%"),
            CompositeStatus::Verifying => "Verifying model…".to_owned(),
            CompositeStatus::EngineFailed => "Engine unavailable".to_owned(),
            CompositeStatus::EngineLoading => "Loading model…".to_owned(),
            CompositeStatus::Recording => "Stop Listening".to_owned(),
            CompositeStatus::Ready => "Start Listening".to_owned(),
        }
    }

    /// Whether the tray toggle / panel's dictate control should accept a
    /// click — [`CompositeStatus::Recording`] (to stop it),
    /// [`CompositeStatus::Ready`] (to start it), and, since F18,
    /// [`CompositeStatus::PermissionsMissing`] too: the Stop control must
    /// stay reachable for a session already running when a permission gets
    /// revoked mid-session (A5, `readiness.rs`), and a fresh attempt should
    /// still be allowed to *try* — if a permission genuinely blocks it,
    /// `ParakeetEngine::start_stream`'s own precheck (or the pipeline's
    /// error path) surfaces that, rather than a disabled toggle silently
    /// refusing to even try. Every other state has a blocking requirement
    /// (relaunch, model, engine failure) that a toggle click can't resolve.
    #[must_use]
    pub(crate) fn toggle_enabled(&self) -> bool {
        matches!(
            self,
            CompositeStatus::Recording
                | CompositeStatus::Ready
                | CompositeStatus::PermissionsMissing
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `StatusModel` in the fully-ready state, so each test only
    /// has to override the one or two fields it cares about.
    fn base_model() -> StatusModel {
        StatusModel {
            model: Some(ModelStatus::Ready),
            engine: EngineState::Ready,
            recording: false,
            hotkey: HotkeyState::Active(HotkeySetting::CapsLock),
            permissions_missing: Vec::new(),
            launch_blocked: false,
            settings_load_warning: None,
        }
    }

    // ── composite() priority table ─────────────────────────────────────

    /// F8/F18's direct regression: a live session outranks a missing
    /// permission (and, transitively, everything else — see
    /// `CompositeStatus`'s own doc comment) — the Stop control must stay
    /// reachable no matter what else is going on.
    #[test]
    fn recording_outranks_every_other_state() {
        let mut model = base_model();
        model.recording = true;
        model.permissions_missing = vec![(Permission::Microphone, Access::Denied)];
        model.launch_blocked = true;
        model.model = Some(ModelStatus::Failed {
            message: "boom".to_owned(),
        });
        model.engine = EngineState::Failed("boom".to_owned());
        assert_eq!(model.composite(), CompositeStatus::Recording);
    }

    #[test]
    fn permissions_missing_outranks_relaunch_model_and_engine() {
        let mut model = base_model();
        model.permissions_missing = vec![(Permission::Microphone, Access::Denied)];
        model.launch_blocked = true;
        model.model = Some(ModelStatus::Failed {
            message: "boom".to_owned(),
        });
        model.engine = EngineState::Failed("boom".to_owned());
        assert_eq!(model.composite(), CompositeStatus::PermissionsMissing);
    }

    #[test]
    fn relaunch_required_when_launch_blocked_and_nothing_missing() {
        let mut model = base_model();
        model.launch_blocked = true;
        assert_eq!(model.composite(), CompositeStatus::RelaunchRequired);
    }

    #[test]
    fn relaunch_required_outranks_model_and_engine_state() {
        let mut model = base_model();
        model.launch_blocked = true;
        model.model = Some(ModelStatus::Missing { total_bytes: 100 });
        model.engine = EngineState::Failed("boom".to_owned());
        assert_eq!(model.composite(), CompositeStatus::RelaunchRequired);
    }

    #[test]
    fn launch_blocked_with_permissions_missing_is_still_permissions_missing() {
        let mut model = base_model();
        model.launch_blocked = true;
        model.permissions_missing = vec![(Permission::Accessibility, Access::Promptable)];
        assert_eq!(model.composite(), CompositeStatus::PermissionsMissing);
    }

    #[test]
    fn model_missing_from_missing_variant() {
        let mut model = base_model();
        model.model = Some(ModelStatus::Missing { total_bytes: 474_000_000 });
        assert_eq!(model.composite(), CompositeStatus::ModelMissing);
    }

    #[test]
    fn model_missing_from_failed_variant() {
        let mut model = base_model();
        model.model = Some(ModelStatus::Failed {
            message: "disk full".to_owned(),
        });
        assert_eq!(model.composite(), CompositeStatus::ModelMissing);
    }

    #[test]
    fn model_missing_outranks_downloading_verifying_and_engine() {
        // Only one `model` field exists, so exercise the priority against
        // engine state directly: a `Missing`/`Failed` model must win over
        // any engine state. `recording` is deliberately left `false` here —
        // since F8/F18, `Recording` outranks everything (including this),
        // so it would defeat the point of this test to set it.
        let mut model = base_model();
        model.model = Some(ModelStatus::Missing { total_bytes: 100 });
        model.engine = EngineState::Failed("boom".to_owned());
        assert_eq!(model.composite(), CompositeStatus::ModelMissing);
    }

    #[test]
    fn downloading_rounds_the_percent() {
        let mut model = base_model();
        model.model = Some(ModelStatus::Downloading {
            received_bytes: 43,
            total_bytes: 100,
        });
        assert_eq!(model.composite(), CompositeStatus::Downloading(43));
    }

    #[test]
    fn downloading_rounds_half_up() {
        let mut model = base_model();
        model.model = Some(ModelStatus::Downloading {
            received_bytes: 1,
            total_bytes: 3,
        });
        // 1/3 = 33.33...% → rounds to 33.
        assert_eq!(model.composite(), CompositeStatus::Downloading(33));
    }

    #[test]
    fn downloading_outranks_verifying_and_engine_state() {
        let mut model = base_model();
        model.model = Some(ModelStatus::Downloading {
            received_bytes: 1,
            total_bytes: 2,
        });
        model.engine = EngineState::Failed("boom".to_owned());
        assert_eq!(model.composite(), CompositeStatus::Downloading(50));
    }

    #[test]
    fn verifying_outranks_engine_state() {
        let mut model = base_model();
        model.model = Some(ModelStatus::Verifying);
        model.engine = EngineState::Failed("boom".to_owned());
        assert_eq!(model.composite(), CompositeStatus::Verifying);
    }

    #[test]
    fn model_ready_contributes_nothing_model_related() {
        let mut model = base_model();
        model.model = Some(ModelStatus::Ready);
        model.engine = EngineState::Loading;
        assert_eq!(model.composite(), CompositeStatus::EngineLoading);
    }

    #[test]
    fn model_none_contributes_nothing_model_related() {
        let mut model = base_model();
        model.model = None;
        model.engine = EngineState::Loading;
        assert_eq!(model.composite(), CompositeStatus::EngineLoading);
    }

    #[test]
    fn recording_outranks_engine_failed() {
        // Since F8/F18, `Recording` outranks every other state (see
        // `CompositeStatus`'s doc comment) — this replaces the old
        // `engine_failed_outranks_recording`, whose expectation was the
        // opposite.
        let mut model = base_model();
        model.engine = EngineState::Failed("boom".to_owned());
        model.recording = true;
        assert_eq!(model.composite(), CompositeStatus::Recording);
    }

    #[test]
    fn recording_outranks_engine_loading() {
        // Replaces the old `engine_loading_outranks_recording` — see
        // `recording_outranks_engine_failed`'s comment.
        let mut model = base_model();
        model.engine = EngineState::Loading;
        model.recording = true;
        assert_eq!(model.composite(), CompositeStatus::Recording);
    }

    #[test]
    fn recording_true_is_recording() {
        let mut model = base_model();
        model.recording = true;
        assert_eq!(model.composite(), CompositeStatus::Recording);
    }

    #[test]
    fn fully_idle_is_ready() {
        assert_eq!(base_model().composite(), CompositeStatus::Ready);
    }

    // ── menu_title() ────────────────────────────────────────────────────

    #[test]
    fn menu_titles_are_distinct_and_nonempty() {
        let all = [
            CompositeStatus::PermissionsMissing,
            CompositeStatus::RelaunchRequired,
            CompositeStatus::ModelMissing,
            CompositeStatus::Downloading(43),
            CompositeStatus::Verifying,
            CompositeStatus::EngineFailed,
            CompositeStatus::EngineLoading,
            CompositeStatus::Recording,
            CompositeStatus::Ready,
        ];
        let titles: Vec<String> = all.iter().map(CompositeStatus::menu_title).collect();
        for title in &titles {
            assert!(!title.is_empty());
        }
        let mut unique = titles.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            titles.len(),
            "menu titles must be distinct: {titles:?}"
        );
    }

    #[test]
    fn relaunch_required_menu_title_is_exact() {
        assert_eq!(
            CompositeStatus::RelaunchRequired.menu_title(),
            "Relaunch required"
        );
    }

    // ── toggle_enabled() totality ───────────────────────────────────────

    #[test]
    fn toggle_enabled_true_exactly_for_recording_ready_and_permissions_missing() {
        let all = [
            CompositeStatus::PermissionsMissing,
            CompositeStatus::RelaunchRequired,
            CompositeStatus::ModelMissing,
            CompositeStatus::Downloading(43),
            CompositeStatus::Verifying,
            CompositeStatus::EngineFailed,
            CompositeStatus::EngineLoading,
            CompositeStatus::Recording,
            CompositeStatus::Ready,
        ];
        for status in all {
            let expected = matches!(
                status,
                CompositeStatus::Recording
                    | CompositeStatus::Ready
                    | CompositeStatus::PermissionsMissing
            );
            assert_eq!(
                status.toggle_enabled(),
                expected,
                "toggle_enabled mismatch for {status:?}"
            );
        }
    }

    // ── idle_headline() ─────────────────────────────────────────────────

    #[test]
    fn idle_headline_hotkey_failed() {
        let mut model = base_model();
        model.hotkey = HotkeyState::Failed(HotkeySetting::CapsLock);
        let (headline, sub) = model.idle_headline();
        assert!(headline.contains("Hotkey inactive"));
        assert!(headline.contains("Settings"));
        assert!(sub.is_none());
    }

    #[test]
    fn idle_headline_ready_uses_the_configured_preset_label_not_a_hardcoded_one() {
        let mut model = base_model();
        model.hotkey = HotkeyState::Active(HotkeySetting::ControlOptionSpace);
        let (headline, _) = model.idle_headline();
        assert!(
            headline.contains(HotkeySetting::ControlOptionSpace.label()),
            "expected preset label {:?} in headline {headline:?}",
            HotkeySetting::ControlOptionSpace.label()
        );
        assert!(
            !headline.contains("CapsLock"),
            "headline must not hardcode a different preset's label: {headline:?}"
        );
    }

    #[test]
    fn idle_headline_downloading_includes_the_percent() {
        let mut model = base_model();
        model.model = Some(ModelStatus::Downloading {
            received_bytes: 43,
            total_bytes: 100,
        });
        let (_, sub) = model.idle_headline();
        let sub = sub.expect("downloading state has a sub-line");
        assert!(sub.contains("43%"), "expected percent in sub-line: {sub:?}");
    }

    #[test]
    fn idle_headline_model_missing_mentions_the_size() {
        let mut model = base_model();
        model.model = Some(ModelStatus::Missing {
            total_bytes: 474_000_000,
        });
        let (_, sub) = model.idle_headline();
        let sub = sub.expect("missing-model state has a sub-line");
        assert!(
            sub.contains(&readiness::format_mb(474_000_000)),
            "expected size in sub-line: {sub:?}"
        );
    }

    /// F7: the idle block's headline for `EngineLoading` must read as
    /// engine warmup ("Getting ready" / "Warming up the speech engine…"),
    /// distinct from the tray's menu title, which intentionally keeps the
    /// older "Loading model…" wording (`CompositeStatus::menu_title`'s own
    /// space is tighter, and the model is in fact already loaded by this
    /// point — only the engine warmup is in progress).
    #[test]
    fn idle_headline_engine_loading_is_distinct_from_menu_title() {
        let mut model = base_model();
        model.engine = EngineState::Loading;
        let (headline, sub) = model.idle_headline();
        assert_eq!(headline.as_ref(), "Getting ready");
        assert_eq!(
            sub.as_ref().map(SharedString::as_ref),
            Some("Warming up the speech engine…")
        );
        assert_eq!(
            CompositeStatus::EngineLoading.menu_title(),
            "Loading model…"
        );
    }
}
