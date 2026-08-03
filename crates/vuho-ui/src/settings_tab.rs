//! The Settings tab view: microphone/hotkey dropdowns, permission rows, and
//! speech-model provisioning — the panel's (`crate::panel::PanelRoot`)
//! Settings tab, and (before launch permissions are all granted) the panel
//! opened on this tab *is* the permission gate (ARCHITECTURE.md ADR-021,
//! amending ADR-016).
//!
//! Reads [`crate::app_status::StatusModel`] for everything status-shaped
//! (model/engine/permissions/hotkey/launch-blocked/settings-load-warning) —
//! this view never re-derives that state itself (e.g. it never calls
//! `readiness::missing_permissions()` directly) — and reads
//! [`vuho_settings::SettingsStore`] fresh on every render for the persisted
//! microphone/hotkey choice.
//!
//! Every side-effecting action is constructor-injected (CONSTITUTION rule
//! 5): the Download/Retry button sends on the `provision_tx` passed into
//! [`SettingsTab::new`], and the live hotkey listener arrives via
//! [`SettingsTab::connect_hotkey`] — this view never reaches through a
//! process-lifetime global to get either.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use crossbeam_channel::Sender;
use gpui::{div, prelude::*, px, Context, Entity, Hsla, IntoElement, Render, SharedString, Window};
use vuho_domain::{DictationCommand, ModelStatus};
use vuho_os_integration::HotkeyListener;
use vuho_settings::{HotkeySetting, SettingsStore};

use crate::app_status::{EngineState, HotkeyState, StatusModel};
use crate::controls;
use crate::hotkey_presets;
use crate::permissions;
use crate::readiness::{self, Access, Permission};
use crate::theme;
use crate::wiring::ProvisionCommand;

/// The Settings tab's root view.
pub(crate) struct SettingsTab {
    status: Entity<StatusModel>,
    settings: Arc<SettingsStore>,
    provision_tx: Sender<ProvisionCommand>,
    /// `None` in gate mode / before production wiring — the hotkey dropdown
    /// still persists a selection then, it just can't live-rebind anything.
    hotkey: Option<Rc<RefCell<HotkeyListener>>>,
    /// `HotkeyListener::start`'s required sender, paired 1:1 with `hotkey`
    /// in production wiring.
    cmd_tx: Option<Sender<DictationCommand>>,
    /// Input device names, snapshotted by [`Self::refresh_devices`] (called
    /// at construction and whenever the microphone dropdown opens).
    devices: Vec<String>,
    mic_open: bool,
    hotkey_open: bool,
}

impl SettingsTab {
    pub(crate) fn new(
        status: Entity<StatusModel>,
        settings: Arc<SettingsStore>,
        provision_tx: Sender<ProvisionCommand>,
        hotkey: Option<Rc<RefCell<HotkeyListener>>>,
        cmd_tx: Option<Sender<DictationCommand>>,
        cx: &mut Context<Self>,
    ) -> Self {
        // Repaint whenever the shared status model changes — the provisioning
        // loop, the permission poll, and this view's own hotkey rebind all
        // write to it independently of this view's own render cycle.
        cx.observe(&status, |_this, _status, cx| cx.notify()).detach();
        let devices = list_devices();
        Self {
            status,
            settings,
            provision_tx,
            hotkey,
            cmd_tx,
            devices,
            mic_open: false,
            hotkey_open: false,
        }
    }

    /// Re-snapshot the input device list — called at construction and again
    /// whenever the microphone dropdown opens, so a device plugged in after
    /// the tab was first shown still appears without needing a reopen.
    pub(crate) fn refresh_devices(&mut self, cx: &mut Context<Self>) {
        self.devices = list_devices();
        cx.notify();
    }

    /// Connect a just-started production hotkey listener into this tab —
    /// called once by `wiring::wire_production`, right after
    /// `wiring::start_hotkey` succeeds, so the hotkey dropdown can
    /// live-rebind it. Never called on the permissions/relaunch-blocked
    /// startup path (`main.rs`) — `hotkey`/`cmd_tx` simply stay `None`
    /// there, and [`Self::select_hotkey`] persists the choice without
    /// rebinding anything.
    pub(crate) fn connect_hotkey(
        &mut self,
        hotkey: Rc<RefCell<HotkeyListener>>,
        cmd_tx: Sender<DictationCommand>,
    ) {
        self.hotkey = Some(hotkey);
        self.cmd_tx = Some(cmd_tx);
    }

    /// Persist the chosen microphone (`None` = system default) and close the
    /// dropdown. Applied at the *next* session start (ADR-013) —
    /// deliberately not live-rebound.
    fn select_microphone(&mut self, choice: Option<String>, cx: &mut Context<Self>) {
        self.mic_open = false;
        if let Err(e) = self.settings.update(|s| s.microphone = choice) {
            log::warn!("settings_tab: failed to save microphone setting: {e}");
        }
        cx.notify();
    }

    /// Persist the chosen hotkey preset, close the dropdown, and — when a
    /// live listener was injected (production mode) — rebind it: `stop()`
    /// then `start()` with the new config, deferring the Accessibility prompt via
    /// `cx.spawn` on failure (same nested-run-loop hazard documented on
    /// `wiring::start_hotkey`). Either way, the resulting `HotkeyState` is
    /// written back into the shared `StatusModel` so the tray/panel and this
    /// tab agree on whether the hotkey is actually listening.
    fn select_hotkey(&mut self, preset: HotkeySetting, cx: &mut Context<Self>) {
        self.hotkey_open = false;

        if let Err(e) = self.settings.update(|s| s.hotkey = preset) {
            log::warn!("settings_tab: failed to save hotkey setting: {e}");
        }

        // Gate mode (no live listener injected): persist only, nothing to
        // rebind or report back to `StatusModel`.
        let (Some(hotkey), Some(cmd_tx)) = (self.hotkey.clone(), self.cmd_tx.clone()) else {
            cx.notify();
            return;
        };

        let start_result = {
            let mut listener = hotkey.borrow_mut();
            listener.stop();
            listener.start(&cmd_tx, hotkey_presets::to_hotkey_config(preset))
        };

        let new_state = if start_result.is_ok() {
            HotkeyState::Active(preset)
        } else {
            HotkeyState::Failed(preset)
        };
        self.status.update(cx, |model, cx| {
            model.hotkey = new_state;
            cx.notify();
        });

        if start_result.is_err() {
            cx.spawn(|_this, _cx: &mut gpui::AsyncApp| async move {
                permissions::prompt_accessibility();
            })
            .detach();
        }
        cx.notify();
    }

    /// The Speech Model section's content, dispatched on `model`/`engine`.
    /// Only ever called when [`should_show_speech_model_section`] is `true`
    /// for the pair — see [`Self::render`].
    fn render_speech_model_section(
        &self,
        model: &ModelStatus,
        engine: &EngineState,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let card = theme::section_card()
            .flex()
            .flex_col()
            .gap_2()
            .child(theme::section_label("Speech Model"));

        match model {
            ModelStatus::Ready => match engine {
                EngineState::Loading => {
                    card.child(model_status_line("Warming up the speech engine…", theme::TEXT_SECONDARY))
                }
                EngineState::Failed(message) => card
                    .child(model_status_line(message, theme::ERROR_RED))
                    .child(self.download_retry_button("Retry", cx)),
                // Unreachable — `should_show_speech_model_section` excludes
                // `Ready`+`Ready` from ever reaching this function.
                EngineState::Ready => card,
            },
            ModelStatus::Missing { .. } => card
                .child(model_status_line(
                    readiness::model_status_text(model),
                    theme::TEXT_SECONDARY,
                ))
                .child(self.download_retry_button("Download", cx)),
            ModelStatus::Downloading {
                received_bytes,
                total_bytes,
            } => card
                .child(theme::progress_bar(model.fraction().unwrap_or(0.0)))
                .child(model_status_line(
                    format!(
                        "{} of {}",
                        readiness::format_mb(*received_bytes),
                        readiness::format_mb(*total_bytes)
                    ),
                    theme::TEXT_SECONDARY,
                ))
                .child(controls::disabled_pill("In progress…")),
            ModelStatus::Verifying => card
                .child(model_status_line("Verifying…", theme::TEXT_SECONDARY))
                .child(controls::disabled_pill("In progress…")),
            ModelStatus::Failed { message } => card
                .child(model_status_line(message, theme::ERROR_RED))
                .child(self.download_retry_button("Retry", cx)),
        }
    }

    /// The Download/Retry button: sends [`ProvisionCommand::Download`] on
    /// the constructor-injected `provision_tx` — never through
    /// `cx.global::<VuhoState>()` (see this module's doc comment). Also what
    /// a `Failed`-engine "Retry" sends: `wiring::on_provision_command`
    /// already dispatches a `Download` command differently depending on the
    /// current `Phase` (re-download vs. retry-the-engine-load-only), so this
    /// view doesn't need to know which one applies.
    fn download_retry_button(&self, label: &'static str, cx: &mut Context<Self>) -> gpui::AnyElement {
        let provision_tx = self.provision_tx.clone();
        controls::action_button(
            label,
            "settings-tab-speech-model-action",
            theme::ACCENT,
            cx.listener(move |_this, _event, _window, _cx| {
                let _ = provision_tx.send(ProvisionCommand::Download);
            }),
        )
        .into_any_element()
    }

    /// The microphone row: label + dropdown, expanding to "System Default" +
    /// every enumerated device (plus a greyed-out entry for a persisted
    /// device that's no longer connected — see [`mic_display`]) when open.
    fn render_mic_row(&self, persisted: Option<&str>, cx: &mut Context<Self>) -> impl IntoElement {
        let display = mic_display(persisted, &self.devices);
        let (label, label_color): (SharedString, Hsla) = match &display {
            MicDisplay::SystemDefault => ("System Default".into(), theme::TEXT_PRIMARY),
            MicDisplay::Connected(name) => (name.clone().into(), theme::TEXT_PRIMARY),
            MicDisplay::Missing(name) => (name.clone().into(), theme::WARN_AMBER),
        };

        let mut column = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(theme::section_label("Microphone"))
            .child(controls::dropdown_button(
                label,
                label_color,
                "settings-tab-mic-dropdown",
                cx.listener(|view, _event, _window, cx| {
                    let opening = !view.mic_open;
                    view.mic_open = opening;
                    view.hotkey_open = false;
                    if opening {
                        view.refresh_devices(cx);
                    } else {
                        cx.notify();
                    }
                }),
            ));

        if let MicDisplay::Missing(_) = &display {
            column = column.child(model_status_line(
                "Not connected — using System Default until it returns.",
                theme::WARN_AMBER,
            ));
        }

        if self.mic_open {
            let mut options = controls::dropdown_option_list();
            options = options.child(controls::dropdown_option(
                "System Default",
                theme::TEXT_PRIMARY,
                ("settings-tab-mic-opt", 0usize),
                cx.listener(|view, _event, _window, cx| view.select_microphone(None, cx)),
            ));
            for (ix, device) in self.devices.iter().cloned().enumerate() {
                let device_for_click = device.clone();
                options = options.child(controls::dropdown_option(
                    device,
                    theme::TEXT_PRIMARY,
                    ("settings-tab-mic-opt", ix + 1),
                    cx.listener(move |view, _event, _window, cx| {
                        view.select_microphone(Some(device_for_click.clone()), cx);
                    }),
                ));
            }
            // The persisted-but-disconnected device (if any): greyed out,
            // still clickable to keep it selected — never cleared
            // automatically (see `mic_display`'s doc comment).
            if let MicDisplay::Missing(name) = &display {
                let name_for_click = name.clone();
                options = options.child(controls::dropdown_option(
                    name.clone(),
                    theme::TEXT_DISABLED,
                    ("settings-tab-mic-opt", self.devices.len() + 1),
                    cx.listener(move |view, _event, _window, cx| {
                        view.select_microphone(Some(name_for_click.clone()), cx);
                    }),
                ));
            }
            column = column.child(options);
        }

        column
    }

    /// The hotkey row: label + dropdown over every [`HotkeySetting`] preset,
    /// plus a persistent error row while the configured preset failed to
    /// bind (`HotkeyState::Failed`).
    fn render_hotkey_row(&self, hotkey_state: HotkeyState, cx: &mut Context<Self>) -> impl IntoElement {
        let mut column = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(theme::section_label("Hotkey"))
            .child(controls::dropdown_button(
                hotkey_label(hotkey_state),
                theme::TEXT_PRIMARY,
                "settings-tab-hotkey-dropdown",
                cx.listener(|view, _event, _window, cx| {
                    view.hotkey_open = !view.hotkey_open;
                    view.mic_open = false;
                    cx.notify();
                }),
            ));

        if self.hotkey_open {
            let mut options = controls::dropdown_option_list();
            for (ix, preset) in HotkeySetting::ALL.into_iter().enumerate() {
                options = options.child(controls::dropdown_option(
                    preset.label(),
                    theme::TEXT_PRIMARY,
                    ("settings-tab-hotkey-opt", ix),
                    cx.listener(move |view, _event, _window, cx| {
                        view.select_hotkey(preset, cx);
                    }),
                ));
            }
            column = column.child(options);
        }

        if matches!(hotkey_state, HotkeyState::Failed(_)) {
            column = column.child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(model_status_line(
                        format!(
                            "Hotkey inactive — Vuho can't listen for {}. Re-grant \
                             Accessibility, then relaunch.",
                            hotkey_label(hotkey_state)
                        ),
                        theme::ERROR_RED,
                    ))
                    .child(controls::action_button(
                        "Relaunch Vuho",
                        "settings-tab-hotkey-relaunch",
                        theme::OK_GREEN,
                        cx.listener(|_view, _event, _window, _cx| readiness::relaunch()),
                    )),
            );
        }

        column
    }
}

impl Render for SettingsTab {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (model, engine, hotkey_state, permissions_missing, launch_blocked, settings_load_warning) = {
            let status = self.status.read(cx);
            (
                status.model.clone(),
                status.engine.clone(),
                status.hotkey,
                status.permissions_missing.clone(),
                status.launch_blocked,
                status.settings_load_warning.clone(),
            )
        };
        let settings = self.settings.get();

        let mut column = div().flex().flex_col().gap_4();

        if let Some(warning) = settings_load_warning {
            column = column.child(render_settings_warning_banner(warning));
        }

        if let Some(model_status) = model.as_ref() {
            if should_show_speech_model_section(Some(model_status), &engine) {
                column = column.child(self.render_speech_model_section(model_status, &engine, cx));
            }
        }

        column = column.child(render_permissions_section(
            &permissions_missing,
            launch_blocked,
            hotkey_state,
            cx,
        ));

        column = column.child(self.render_mic_row(settings.microphone.as_deref(), cx));
        column = column.child(self.render_hotkey_row(hotkey_state, cx));

        column
    }
}

// ── Pure helpers (unit-tested without GPUI) ─────────────────────────────

/// Re-snapshot input device names from `vuho_stt_engine`. An enumeration
/// failure just means the dropdown offers only "System Default".
fn list_devices() -> Vec<String> {
    match vuho_stt_engine::list_input_devices() {
        Ok(devices) => devices,
        Err(e) => {
            log::warn!("settings_tab: failed to list input devices: {e}");
            Vec::new()
        }
    }
}

/// Whether the Speech Model section should render at all: only when a model
/// status has been observed (`None` means gate mode, which never reaches
/// provisioning) and the pair isn't already fully settled (`Ready` model +
/// `Ready` engine — nothing left to say).
#[must_use]
fn should_show_speech_model_section(model: Option<&ModelStatus>, engine: &EngineState) -> bool {
    match model {
        Some(status) => !(*status == ModelStatus::Ready && *engine == EngineState::Ready),
        None => false,
    }
}

/// The label to show for a hotkey state — the preset it carries either way,
/// whether currently active or failed to bind (both need a label to render).
#[must_use]
fn hotkey_label(state: HotkeyState) -> &'static str {
    match state {
        HotkeyState::Active(preset) | HotkeyState::Failed(preset) => preset.label(),
    }
}

/// One permission row's rendered shape, derived purely from its live
/// [`Access`] plus two booleans the caller already knows: whether this row
/// is Accessibility (the one permission with a relaunch-after-grant wrinkle)
/// and whether the configured hotkey currently fails to bind. Free-standing
/// and pure so it's unit-testable without GPUI or real TCC state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowKind {
    /// Granted, nothing else to say.
    Granted,
    /// Accessibility specifically: granted, but the hotkey still failed to
    /// bind (a grant made *after* the failed `start()` needs a relaunch to
    /// take effect for a `CGEventTap`, not just a re-select in this UI).
    GrantedNeedsRelaunch,
    /// Never asked — the native one-click prompt will work.
    Promptable,
    /// Already answered no (or MDM-restricted) — only System Settings can
    /// change it now.
    Denied,
}

#[must_use]
fn permission_row_kind(access: Access, is_accessibility: bool, hotkey_failed: bool) -> RowKind {
    match access {
        Access::Granted if is_accessibility && hotkey_failed => RowKind::GrantedNeedsRelaunch,
        Access::Granted => RowKind::Granted,
        Access::Promptable => RowKind::Promptable,
        Access::Denied => RowKind::Denied,
    }
}

/// How the microphone dropdown should render the persisted device choice —
/// pure so the "is the persisted device still connected" decision is
/// unit-testable without `cpal`/real hardware.
///
/// Never clears the persisted setting by itself (see [`SettingsTab::render_mic_row`]'s
/// "still clickable to keep" handling) — a device that's merely unplugged
/// (Bluetooth headset out of range, a dock disconnected) should still be
/// there, greyed out, the next time it reconnects.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MicDisplay {
    SystemDefault,
    /// The persisted device name, currently enumerated.
    Connected(String),
    /// The persisted device name, NOT currently enumerated — Vuho falls
    /// back to System Default until it returns.
    Missing(String),
}

#[must_use]
fn mic_display(persisted: Option<&str>, devices: &[String]) -> MicDisplay {
    match persisted {
        None => MicDisplay::SystemDefault,
        Some(name) => {
            if devices.iter().any(|d| d == name) {
                MicDisplay::Connected(name.to_owned())
            } else {
                MicDisplay::Missing(name.to_owned())
            }
        }
    }
}

// ── Rendering helpers (CONSTITUTION rule 28: split, ≤40 lines each) ───────

/// The settings-load-warning banner: only rendered when
/// `StatusModel::settings_load_warning` is `Some`.
fn render_settings_warning_banner(detail: SharedString) -> impl IntoElement {
    theme::section_card()
        .bg(theme::WARN_AMBER.opacity(0.12))
        .flex()
        .flex_col()
        .gap_1()
        .child(model_status_line(
            "Settings could not be read — defaults are in use.",
            theme::WARN_AMBER,
        ))
        .child(model_status_line(detail, theme::TEXT_SECONDARY))
}

/// A single line of status text at [`theme::TEXT_SM`], in the given color —
/// the one shape every section below uses for its status/error/warning line.
fn model_status_line(text: impl Into<SharedString>, color: Hsla) -> impl IntoElement {
    div()
        .text_size(px(theme::TEXT_SM))
        .text_color(color)
        .child(text.into())
}

/// The Permissions section: a compact "all granted" row, or one row per
/// [`Permission::ALL`] plus (independently) the launch-blocked relaunch card
/// — see this module's doc comment on why those two conditions don't
/// exclude each other.
fn render_permissions_section(
    permissions_missing: &[Permission],
    launch_blocked: bool,
    hotkey: HotkeyState,
    cx: &mut Context<SettingsTab>,
) -> impl IntoElement {
    let mut card = theme::section_card()
        .flex()
        .flex_col()
        .gap_2()
        .child(theme::section_label("Permissions"));

    if permissions_missing.is_empty() && !launch_blocked {
        card = card.child(render_all_granted_row());
    } else {
        let hotkey_failed = matches!(hotkey, HotkeyState::Failed(_));
        for permission in Permission::ALL {
            card = card.child(render_permission_row(permission, hotkey_failed, cx));
        }
    }

    if launch_blocked && permissions_missing.is_empty() {
        card = card.child(render_relaunch_gate_row(cx));
    }

    card
}

/// The compact "everything's fine" permissions row: a green check + one line
/// of text, no buttons.
fn render_all_granted_row() -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap_2()
        .child(div().text_color(theme::OK_GREEN).child("✓"))
        .child(model_status_line(
            "All permissions granted",
            theme::TEXT_SECONDARY,
        ))
}

/// One permission row: label, description, an Accessibility-only relaunch
/// note, and an action driven by [`permission_row_kind`].
fn render_permission_row(
    permission: Permission,
    hotkey_failed: bool,
    cx: &mut Context<SettingsTab>,
) -> impl IntoElement {
    let access = permission.access();
    let is_accessibility = matches!(permission, Permission::Accessibility);
    let kind = permission_row_kind(access, is_accessibility, hotkey_failed);

    let mut row = div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .text_size(px(theme::TEXT_MD))
                .text_color(theme::TEXT_PRIMARY)
                .child(permission.label()),
        )
        .child(model_status_line(permission.description(), theme::TEXT_TERTIARY));

    if is_accessibility && access != Access::Granted {
        row = row.child(model_status_line(
            "Vuho must be relaunched after granting.",
            theme::TEXT_TERTIARY,
        ));
    }

    row = match kind {
        RowKind::Granted => {
            row.child(model_status_line("✓ Granted", theme::OK_GREEN))
        }
        RowKind::GrantedNeedsRelaunch => row
            .child(model_status_line(
                "Granted — relaunch required",
                theme::WARN_AMBER,
            ))
            .child(controls::action_button(
                "Relaunch Vuho",
                ("settings-tab-permission-relaunch", permission as usize),
                theme::OK_GREEN,
                cx.listener(|_view, _event, _window, _cx| readiness::relaunch()),
            )),
        RowKind::Promptable => row.child(controls::action_button(
            format!("Allow {}", permission.label()),
            ("settings-tab-permission-allow", permission as usize),
            theme::ACCENT,
            cx.listener(move |_view, _event, _window, _cx| permission.request()),
        )),
        RowKind::Denied => row
            .child(model_status_line("Access denied", theme::ERROR_RED))
            .child(controls::action_button(
                "Open System Settings",
                ("settings-tab-permission-settings", permission as usize),
                theme::ACCENT,
                cx.listener(move |_view, _event, _window, _cx| {
                    permissions::open_url(permission.settings_url());
                }),
            )),
    };

    row
}

/// The launch-blocked relaunch card: shown whenever every permission is
/// granted but the process itself still needs a relaunch (a TCC grant is a
/// process-identity fact) — never auto-hidden by anything else in this
/// section.
fn render_relaunch_gate_row(cx: &mut Context<SettingsTab>) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .p_3()
        .rounded(px(theme::RADIUS_CARD))
        .bg(theme::FILL_SELECTED)
        .child(model_status_line(
            "All permissions granted — relaunch Vuho to finish setup.",
            theme::TEXT_PRIMARY,
        ))
        .child(controls::action_button(
            "Relaunch Vuho",
            "settings-tab-relaunch-gate",
            theme::OK_GREEN,
            cx.listener(|_view, _event, _window, _cx| readiness::relaunch()),
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── permission_row_kind ──────────────────────────────────────────────

    #[test]
    fn granted_non_accessibility_is_granted() {
        assert_eq!(
            permission_row_kind(Access::Granted, false, true),
            RowKind::Granted
        );
    }

    #[test]
    fn granted_accessibility_with_working_hotkey_is_granted() {
        assert_eq!(
            permission_row_kind(Access::Granted, true, false),
            RowKind::Granted
        );
    }

    #[test]
    fn granted_accessibility_with_failed_hotkey_needs_relaunch() {
        assert_eq!(
            permission_row_kind(Access::Granted, true, true),
            RowKind::GrantedNeedsRelaunch
        );
    }

    #[test]
    fn hotkey_failed_only_matters_for_accessibility() {
        // A non-Accessibility permission never surfaces the relaunch state,
        // even if the hotkey happens to be failed for an unrelated reason.
        assert_eq!(
            permission_row_kind(Access::Granted, false, true),
            RowKind::Granted
        );
    }

    #[test]
    fn promptable_is_promptable_regardless_of_accessibility_or_hotkey() {
        for is_accessibility in [false, true] {
            for hotkey_failed in [false, true] {
                assert_eq!(
                    permission_row_kind(Access::Promptable, is_accessibility, hotkey_failed),
                    RowKind::Promptable
                );
            }
        }
    }

    #[test]
    fn denied_is_denied_regardless_of_accessibility_or_hotkey() {
        for is_accessibility in [false, true] {
            for hotkey_failed in [false, true] {
                assert_eq!(
                    permission_row_kind(Access::Denied, is_accessibility, hotkey_failed),
                    RowKind::Denied
                );
            }
        }
    }

    // ── mic_display ──────────────────────────────────────────────────────

    #[test]
    fn no_persisted_device_is_system_default() {
        assert_eq!(mic_display(None, &[]), MicDisplay::SystemDefault);
        assert_eq!(
            mic_display(None, &["Built-in Mic".to_owned()]),
            MicDisplay::SystemDefault
        );
    }

    #[test]
    fn persisted_device_present_in_list_is_connected() {
        let devices = vec!["USB Mic".to_owned(), "Built-in Mic".to_owned()];
        assert_eq!(
            mic_display(Some("Built-in Mic"), &devices),
            MicDisplay::Connected("Built-in Mic".to_owned())
        );
    }

    #[test]
    fn persisted_device_absent_from_list_is_missing() {
        let devices = vec!["USB Mic".to_owned()];
        assert_eq!(
            mic_display(Some("Bluetooth Headset"), &devices),
            MicDisplay::Missing("Bluetooth Headset".to_owned())
        );
    }

    #[test]
    fn persisted_device_missing_from_an_empty_list_is_missing() {
        assert_eq!(
            mic_display(Some("USB Mic"), &[]),
            MicDisplay::Missing("USB Mic".to_owned())
        );
    }

    // ── should_show_speech_model_section ────────────────────────────────

    #[test]
    fn hidden_when_model_is_none() {
        assert!(!should_show_speech_model_section(None, &EngineState::Ready));
        assert!(!should_show_speech_model_section(
            None,
            &EngineState::Loading
        ));
    }

    #[test]
    fn hidden_when_model_and_engine_are_both_ready() {
        assert!(!should_show_speech_model_section(
            Some(&ModelStatus::Ready),
            &EngineState::Ready
        ));
    }

    #[test]
    fn shown_when_model_ready_but_engine_is_not() {
        assert!(should_show_speech_model_section(
            Some(&ModelStatus::Ready),
            &EngineState::Loading
        ));
        assert!(should_show_speech_model_section(
            Some(&ModelStatus::Ready),
            &EngineState::Failed("boom".to_owned())
        ));
    }

    #[test]
    fn shown_when_model_is_not_ready_regardless_of_engine() {
        let missing = ModelStatus::Missing { total_bytes: 100 };
        assert!(should_show_speech_model_section(
            Some(&missing),
            &EngineState::Ready
        ));
    }

    // ── hotkey_label ─────────────────────────────────────────────────────

    #[test]
    fn hotkey_label_reads_the_carried_preset_either_way() {
        assert_eq!(
            hotkey_label(HotkeyState::Active(HotkeySetting::CapsLock)),
            HotkeySetting::CapsLock.label()
        );
        assert_eq!(
            hotkey_label(HotkeyState::Failed(HotkeySetting::OptionSpace)),
            HotkeySetting::OptionSpace.label()
        );
    }
}
