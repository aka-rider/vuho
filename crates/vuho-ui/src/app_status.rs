//! `StatusModel`: the single source of truth for app state (ADR — UI
//! rehaul), feeding the tray icon/menu, the panel's in-window status
//! display, and (a later package) a Settings tab.
//!
//! [`StatusModel::composite`]/[`CompositeStatus::menu_title`]/
//! [`CompositeStatus::toggle_enabled`] drive `status_bar.rs`'s tray as of
//! the WP4 wiring (`wiring::wire_production`/`event_loop.rs`); the panel
//! that will read [`StatusModel::idle_headline`] and the still-unwired
//! `permissions_missing`/`launch_blocked`/`settings_load_warning` fields is
//! a later package, so those stay behind item-level `#[allow(dead_code)]`.
//!
//! `StatusModel` itself stays a plain struct — no `Global` impl, no
//! channels — so it composes as a GPUI `Entity` (`cx.new(|_| StatusModel {
//! .. })`) exactly like `ReadinessView`/`SettingsView` do today.

use gpui::SharedString;
use vuho_domain::ModelStatus;
use vuho_settings::HotkeySetting;

use crate::readiness::{self, Permission};

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
    pub permissions_missing: Vec<Permission>,
    /// `true` when the app started on the permission-gate path — a relaunch
    /// is required after granting for the new process identity to carry
    /// the TCC grant (see `readiness.rs`'s module doc comment).
    pub launch_blocked: bool,
    /// Consumer is a later package (the panel's settings tab surfacing a
    /// malformed-settings-file warning) — not read anywhere yet.
    #[allow(dead_code, reason = "no reader until the panel package")]
    pub settings_load_warning: Option<SharedString>,
}

/// The one composite state derived from every [`StatusModel`] field, in
/// strict priority order (highest first): a higher-priority condition being
/// true always wins, regardless of what a lower-priority field says —
/// e.g. a `Recording` session with a missing permission still reports
/// [`CompositeStatus::PermissionsMissing`], since a permission revoked
/// mid-session (A5, `readiness.rs`) is more urgent than the fact that a
/// session happens to be running.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum CompositeStatus {
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
    Recording,
    Ready,
}

impl StatusModel {
    /// Derive the one composite state per the priority order documented on
    /// [`CompositeStatus`].
    #[must_use]
    pub(crate) fn composite(&self) -> CompositeStatus {
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
        if self.recording {
            CompositeStatus::Recording
        } else {
            CompositeStatus::Ready
        }
    }

    /// Headline + optional sub-line for the panel's idle status block, one
    /// per [`CompositeStatus`] — computed via [`StatusModel::composite`] so
    /// this never drifts from the priority order it defines.
    ///
    /// No caller until the panel package renders it.
    #[allow(dead_code, reason = "no caller until the panel package")]
    #[must_use]
    pub(crate) fn idle_headline(&self) -> (SharedString, Option<SharedString>) {
        match self.composite() {
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
            CompositeStatus::EngineLoading => (SharedString::from("Loading model…"), None),
            CompositeStatus::Recording => (SharedString::from("Listening…"), None),
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
///
/// Only reachable via [`StatusModel::idle_headline`], which has no caller
/// until the panel package.
#[allow(dead_code, reason = "only reachable via idle_headline")]
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
///
/// Only reachable via [`StatusModel::idle_headline`], which has no caller
/// until the panel package.
#[allow(dead_code, reason = "only reachable via idle_headline")]
fn engine_failed_headline(engine: &EngineState) -> (SharedString, Option<SharedString>) {
    let sub = match engine {
        EngineState::Failed(message) => Some(SharedString::from(message.clone())),
        EngineState::Loading | EngineState::Ready => None,
    };
    (SharedString::from("Engine unavailable"), sub)
}

/// [`CompositeStatus::Ready`]'s headline: an inactive-hotkey warning, or
/// "press `<label>` to dictate" with the label read from the existing
/// [`HotkeySetting::label`] source (`settings_window.rs`/`hotkey_presets.rs`
/// render the same presets today) — never hardcoded, since the preset is
/// user-configurable.
///
/// Only reachable via [`StatusModel::idle_headline`], which has no caller
/// until the panel package.
#[allow(dead_code, reason = "only reachable via idle_headline")]
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
    /// click — only once an engine exists to drive ([`CompositeStatus::Recording`]
    /// to stop it, [`CompositeStatus::Ready`] to start it). Every other
    /// state has a blocking requirement (permission, relaunch, model,
    /// engine failure) that a toggle click can't resolve.
    #[must_use]
    pub(crate) fn toggle_enabled(&self) -> bool {
        matches!(self, CompositeStatus::Recording | CompositeStatus::Ready)
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

    #[test]
    fn permissions_missing_outranks_everything_else() {
        let mut model = base_model();
        model.permissions_missing = vec![Permission::Microphone];
        model.launch_blocked = true;
        model.model = Some(ModelStatus::Failed {
            message: "boom".to_owned(),
        });
        model.engine = EngineState::Failed("boom".to_owned());
        model.recording = true;
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
        model.permissions_missing = vec![Permission::Accessibility];
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
        // any engine state.
        let mut model = base_model();
        model.model = Some(ModelStatus::Missing { total_bytes: 100 });
        model.engine = EngineState::Failed("boom".to_owned());
        model.recording = true;
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
    fn engine_failed_outranks_recording() {
        let mut model = base_model();
        model.engine = EngineState::Failed("boom".to_owned());
        model.recording = true;
        assert_eq!(model.composite(), CompositeStatus::EngineFailed);
    }

    #[test]
    fn engine_loading_outranks_recording() {
        let mut model = base_model();
        model.engine = EngineState::Loading;
        model.recording = true;
        assert_eq!(model.composite(), CompositeStatus::EngineLoading);
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
    fn toggle_enabled_true_exactly_for_recording_and_ready() {
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
            let expected = matches!(status, CompositeStatus::Recording | CompositeStatus::Ready);
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
}
