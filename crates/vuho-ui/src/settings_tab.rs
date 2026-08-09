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
use gpui::{
    div, prelude::*, px, Context, Div, Entity, Hsla, IntoElement, Render, SharedString, Window,
};
use vuho_domain::{DictationCommand, ModelStatus};
use vuho_model_fetch::ModelAvailability;
use vuho_model_paths::{Backend, ModelSource};
use vuho_os_integration::HotkeyListener;
use vuho_settings::{HotkeySetting, SettingsStore};

use crate::app_status::{EngineState, HotkeyState, StatusModel};
use crate::controls;
use crate::hotkey_presets;
use crate::permissions;
use crate::readiness::{self, Access, Permission};
use crate::theme;
use crate::wiring::ProvisionCommand;

/// A live, production-only hotkey listener + the command sender
/// `HotkeyListener::start` requires — paired 1:1 by construction (F22).
/// Before this type existed, the listener and its sender were two
/// independently-`Option`al fields on [`SettingsTab`], which allowed a
/// half-set state (a listener with no sender, or vice versa) that nothing
/// ever legitimately produced — this makes that state unrepresentable.
#[derive(Clone)]
struct LiveHotkey {
    listener: Rc<RefCell<HotkeyListener>>,
    cmd_tx: Sender<DictationCommand>,
}

/// The Settings tab's root view.
pub(crate) struct SettingsTab {
    status: Entity<StatusModel>,
    settings: Arc<SettingsStore>,
    provision_tx: Sender<ProvisionCommand>,
    /// `None` in gate mode / before production wiring — the hotkey dropdown
    /// still persists a selection then, it just can't live-rebind anything.
    /// `Some` only via [`Self::connect_hotkey`], never at construction.
    hotkey: Option<LiveHotkey>,
    /// Input device names, snapshotted by [`Self::refresh_devices`] (called
    /// at construction and whenever the microphone dropdown opens).
    devices: Vec<String>,
    mic_open: bool,
    hotkey_open: bool,
    model_open: bool,
}

impl SettingsTab {
    pub(crate) fn new(
        status: Entity<StatusModel>,
        settings: Arc<SettingsStore>,
        provision_tx: Sender<ProvisionCommand>,
        cx: &mut Context<Self>,
    ) -> Self {
        // Repaint whenever the shared status model changes — the provisioning
        // loop, the permission poll, and this view's own hotkey rebind all
        // write to it independently of this view's own render cycle.
        cx.observe(&status, |_this, _status, cx| cx.notify())
            .detach();
        let devices = list_devices();
        Self {
            status,
            settings,
            provision_tx,
            hotkey: None,
            devices,
            mic_open: false,
            hotkey_open: false,
            model_open: false,
        }
    }

    /// Re-snapshot the input device list — called at construction and again
    /// whenever the microphone dropdown opens, so a device plugged in after
    /// the tab was first shown still appears without needing a reopen.
    pub(crate) fn refresh_devices(&mut self, cx: &mut Context<Self>) {
        self.devices = list_devices();
        cx.notify();
    }

    /// Close both dropdowns without touching anything else (G6) — called by
    /// `panel::hide_root` so dismissing the panel mid-selection doesn't
    /// silently leave a dropdown open, which would otherwise show a
    /// microphone list that's gone stale by the time the panel reopens
    /// (`refresh_devices` only re-snapshots when a dropdown is opened, and
    /// a still-open dropdown never re-opens). No `cx.notify()` here — the
    /// panel is being hidden, not re-rendered, and the next legitimate
    /// render (`show`'s own `refresh_devices` call on the Settings tab, or
    /// any other `StatusModel`-driven repaint) already picks up the closed
    /// state.
    pub(crate) fn close_dropdowns(&mut self) {
        self.mic_open = false;
        self.hotkey_open = false;
        self.model_open = false;
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
        listener: Rc<RefCell<HotkeyListener>>,
        cmd_tx: Sender<DictationCommand>,
    ) {
        self.hotkey = Some(LiveHotkey { listener, cmd_tx });
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
    /// then `start()` with the new config, prompting for Accessibility on
    /// failure (`permissions::prompt_accessibility` is self-deferring, so no
    /// deferral wrapper is needed here). Either way, the resulting
    /// `HotkeyState` is written back into the shared `StatusModel` so the
    /// tray/panel and this tab agree on whether the hotkey is actually
    /// listening.
    fn select_hotkey(&mut self, preset: HotkeySetting, cx: &mut Context<Self>) {
        self.hotkey_open = false;

        if let Err(e) = self.settings.update(|s| s.hotkey = preset) {
            log::warn!("settings_tab: failed to save hotkey setting: {e}");
        }

        // Gate mode (no live listener injected): persist only, nothing to
        // rebind or report back to `StatusModel`.
        let Some(live) = self.hotkey.clone() else {
            cx.notify();
            return;
        };

        let start_result = {
            let mut listener = live.listener.borrow_mut();
            listener.stop();
            listener.start(&live.cmd_tx, hotkey_presets::to_hotkey_config(preset))
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
            permissions::prompt_accessibility();
        }
        cx.notify();
    }

    /// The Speech Model card: the model combobox, the languages the chosen
    /// backend can actually reach, one row per known model, and — below the
    /// list — the engine's own warmup state, which is a property of the
    /// selected model rather than of the list.
    fn render_speech_model_section(
        &self,
        models: &[ModelAvailability],
        engine: &EngineState,
        recording: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let selected = crate::wiring::selected_model_id(&self.settings);
        let mut card = theme::section_card()
            .flex()
            .flex_col()
            .gap_2()
            .child(theme::section_label("Speech Model"))
            .child(self.render_model_row(models, &selected, recording, cx))
            .child(model_status_line(
                languages_line(&selected),
                theme::TEXT_TERTIARY,
            ));

        for (ix, model) in models.iter().enumerate() {
            card = card.child(self.render_model_list_row(model, ix, model.id == selected, cx));
        }
        self.append_engine_state(card, engine, &selected, cx)
    }

    /// The model combobox (WP9.S2): the selected model's display name, and
    /// — while open — one option per known model, each unselectable unless
    /// it is both `Ready` and supported by this macOS. Locked while a
    /// session is recording: swapping the engine out from under a live
    /// dictation would drop the transcript in flight.
    fn render_model_row(
        &self,
        models: &[ModelAvailability],
        selected: &str,
        recording: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let label = model_display_name(models, selected);
        if recording {
            return div().child(controls::disabled_pill(label));
        }

        let mut column = div().flex().flex_col().child(controls::dropdown_button(
            label,
            theme::TEXT_PRIMARY,
            "settings-tab-model-dropdown",
            cx.listener(|view, _event, _window, cx| {
                view.model_open = !view.model_open;
                view.mic_open = false;
                view.hotkey_open = false;
                cx.notify();
            }),
        ));

        if self.model_open {
            column = column.child(render_model_options(models, cx));
        }
        column
    }

    /// Send [`ProvisionCommand::SelectModel`] and close the dropdown. The
    /// choice is persisted by the provisioning thread, which owns both the
    /// settings write and the engine reload it implies — this view never
    /// half-applies a model switch by writing the setting itself.
    fn select_model(&mut self, id: String, cx: &mut Context<Self>) {
        self.model_open = false;
        let _ = self.provision_tx.send(ProvisionCommand::SelectModel(id));
        cx.notify();
    }

    /// One row of the model list: the model's name and size, plus the one
    /// control its state calls for ([`model_row_control`]).
    fn render_model_list_row(
        &self,
        model: &ModelAvailability,
        ix: usize,
        is_selected: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let row = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_size(px(theme::TEXT_MD))
                    .text_color(theme::TEXT_PRIMARY)
                    .child(SharedString::from(model.display_name.clone())),
            )
            .child(model_status_line(
                readiness::format_mb(model.total_bytes),
                theme::TEXT_TERTIARY,
            ));
        self.append_row_control(
            row,
            &model.id,
            model_row_control(model, is_selected),
            ix,
            cx,
        )
    }

    /// Append a model row's [`ModelRowControl`] — one arm per control, with
    /// the multi-child download arm delegated to [`append_download_progress`]
    /// (CONSTITUTION rule 28).
    fn append_row_control(
        &self,
        row: Div,
        id: &str,
        control: ModelRowControl,
        ix: usize,
        cx: &mut Context<Self>,
    ) -> Div {
        match control {
            ModelRowControl::Unsupported(label) | ModelRowControl::Provisioned(label) => {
                row.child(controls::disabled_pill(label))
            }
            ModelRowControl::Selected => row.child(controls::disabled_pill("Selected")),
            ModelRowControl::Download => {
                row.child(self.provision_button("Download", id, ix, theme::ACCENT, download_of, cx))
            }
            ModelRowControl::Downloading {
                received_bytes,
                total_bytes,
            } => append_download_progress(row, received_bytes, total_bytes),
            ModelRowControl::Verifying => row
                .child(model_status_line("Verifying…", theme::TEXT_SECONDARY))
                .child(controls::disabled_pill("In progress…")),
            ModelRowControl::Failed(message) => row
                .child(model_status_line(message, theme::ERROR_RED))
                .child(self.provision_button("Retry", id, ix, theme::ACCENT, download_of, cx)),
            ModelRowControl::Delete => {
                row.child(self.provision_button("Delete", id, ix, theme::ERROR_RED, delete_of, cx))
            }
        }
    }

    /// A model row's action button: sends the [`ProvisionCommand`] `command`
    /// builds for this row's model on the constructor-injected
    /// `provision_tx` — never through a process-lifetime global (see this
    /// module's doc comment).
    fn provision_button(
        &self,
        label: &'static str,
        id: &str,
        ix: usize,
        bg: Hsla,
        command: fn(String) -> ProvisionCommand,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let provision_tx = self.provision_tx.clone();
        let id = id.to_owned();
        controls::action_button(
            label,
            ("settings-tab-model-action", ix),
            bg,
            cx.listener(move |_this, _event, _window, _cx| {
                let _ = provision_tx.send(command(id.clone()));
            }),
        )
        .into_any_element()
    }

    /// The engine's own state, below the model list: warming up, or a load
    /// failure with a Retry that re-selects the already-selected model —
    /// which re-attempts the engine load alone, never a redundant
    /// re-download (see `wiring::Phase::EngineFailed`).
    fn append_engine_state(
        &self,
        card: Div,
        engine: &EngineState,
        selected: &str,
        cx: &mut Context<Self>,
    ) -> Div {
        match engine {
            EngineState::Ready => card,
            EngineState::Loading => card.child(model_status_line(
                "Warming up the speech engine…",
                theme::TEXT_SECONDARY,
            )),
            EngineState::Failed(message) => {
                let provision_tx = self.provision_tx.clone();
                let id = selected.to_owned();
                card.child(model_status_line(message.clone(), theme::ERROR_RED))
                    .child(controls::action_button(
                        "Retry",
                        "settings-tab-engine-retry",
                        theme::ACCENT,
                        cx.listener(move |_this, _event, _window, _cx| {
                            let _ = provision_tx.send(ProvisionCommand::SelectModel(id.clone()));
                        }),
                    ))
            }
        }
    }

    /// The microphone row: label + dropdown, expanding to "System Default" +
    /// every enumerated device (plus a greyed-out entry for a persisted
    /// device that's no longer connected — see [`mic_display`]) when open.
    fn render_mic_row(&self, persisted: Option<&str>, cx: &mut Context<Self>) -> impl IntoElement {
        let display = mic_display(persisted, &self.devices);
        let (label, label_color) = mic_label(&display);

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
                    view.model_open = false;
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
            column = column.child(self.render_mic_options(&display, cx));
        }

        column
    }

    /// The mic dropdown's expanded option list: "System Default", every
    /// enumerated device, and — if any — the persisted-but-disconnected
    /// device, greyed out but still clickable (never cleared automatically,
    /// see [`mic_display`]'s doc comment). Split out of
    /// [`Self::render_mic_row`] to keep it under the 40-line render-helper
    /// limit (CONSTITUTION rule 28).
    fn render_mic_options(&self, display: &MicDisplay, cx: &mut Context<Self>) -> impl IntoElement {
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
        if let MicDisplay::Missing(name) = display {
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
        options
    }

    /// The hotkey row: label + dropdown over every [`HotkeySetting`] preset,
    /// plus a persistent error row while the configured preset failed to
    /// bind (`HotkeyState::Failed`).
    fn render_hotkey_row(
        &self,
        hotkey_state: HotkeyState,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
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
                    view.model_open = false;
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
        let (
            models,
            engine,
            recording,
            hotkey_state,
            permissions_missing,
            launch_blocked,
            settings_load_warning,
            show_speech_model,
        ) = {
            let status = self.status.read(cx);
            (
                status.models.clone(),
                status.engine.clone(),
                status.recording,
                status.hotkey,
                status.permissions_missing.clone(),
                status.launch_blocked,
                status.settings_load_warning.clone(),
                shows_speech_model_section(status),
            )
        };
        let settings = self.settings.get();

        let mut column = div().flex().flex_col().gap_4();

        if let Some(warning) = settings_load_warning {
            column = column.child(render_settings_warning_banner(warning));
        }

        if show_speech_model {
            column =
                column.child(self.render_speech_model_section(&models, &engine, recording, cx));
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

/// Whether the Speech Model card has anything true to show. The whole card
/// renders from the provisioning thread's `UiCommand::ModelList`: with no
/// rows the combobox falls back to the bare model id, the list is empty, and
/// the engine line claims a warmup nobody started — and on the
/// permission-gate startup path that thread never runs at all (ADR-021), so
/// the Download/Delete/Select buttons would send into a dropped channel for
/// the whole process lifetime.
#[must_use]
fn shows_speech_model_section(status: &StatusModel) -> bool {
    !status.models.is_empty()
}

/// Whether the model combobox may offer `model` at all: selecting a model
/// that is not `Ready` would only fail the load, and one this macOS is too
/// old for cannot run at all (WP8.S3 refuses both on the receiving end —
/// this is the same rule stated where the user can see it).
#[must_use]
fn selectable(model: &ModelAvailability) -> bool {
    model.status == ModelStatus::Ready && model.supported_on_this_os
}

/// The combobox's current value: the selected model's display name, falling
/// back to its bare id before the provisioning thread's first
/// `UiCommand::ModelList` has arrived.
#[must_use]
fn model_display_name(models: &[ModelAvailability], selected: &str) -> SharedString {
    models
        .iter()
        .find(|model| model.id == selected)
        .map_or_else(
            || SharedString::from(selected.to_owned()),
            |model| SharedString::from(model.display_name.clone()),
        )
}

/// The languages the selected backend can actually dictate in: the codes the
/// OS layer can report, narrowed to what the backend accepts. Both sets come
/// from their owners (`vuho_os_integration::mapped_languages` and Canary's
/// own prompt table) — never a second copy here (CONSTITUTION rule 26).
#[must_use]
fn languages_line(model_id: &str) -> String {
    let codes = reachable_languages(model_id);
    format!("Languages: {}", codes.join(", "))
}

#[must_use]
fn reachable_languages(model_id: &str) -> Vec<&'static str> {
    let mapped = vuho_os_integration::mapped_languages();
    let backend = vuho_model_paths::manifest()
        .stt
        .model(model_id)
        .map(|model| model.backend);
    let mut codes: Vec<&'static str> = match backend {
        Some(Backend::CanaryAed) => vuho_stt_engine::canary::prompt::supported_languages()
            .filter(|code| mapped.contains(code))
            .collect(),
        Some(Backend::ParakeetTdt) | None => mapped.to_vec(),
    };
    codes.sort_unstable();
    codes
}

/// The single trailing control a model row offers, derived purely from that
/// model's availability and whether it is the selected one — pure so the
/// whole table is unit-testable without GPUI.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelRowControl {
    /// This macOS is older than the model's manifest floor.
    Unsupported(SharedString),
    Download,
    Downloading {
        received_bytes: u64,
        total_bytes: u64,
    },
    Verifying,
    Failed(SharedString),
    /// Ready and in use — nothing to do to it.
    Selected,
    /// Ready, unselected, and Vuho's own download to remove (ADR-020).
    Delete,
    /// Ready and unselected, but provisioned by someone else — the label
    /// says who, since the absence of a Delete button otherwise reads as a
    /// bug.
    Provisioned(SharedString),
}

#[must_use]
fn model_row_control(model: &ModelAvailability, is_selected: bool) -> ModelRowControl {
    if !model.supported_on_this_os {
        return ModelRowControl::Unsupported(min_macos_label(&model.id));
    }
    match &model.status {
        ModelStatus::Missing { .. } => ModelRowControl::Download,
        ModelStatus::Downloading {
            received_bytes,
            total_bytes,
        } => ModelRowControl::Downloading {
            received_bytes: *received_bytes,
            total_bytes: *total_bytes,
        },
        ModelStatus::Verifying => ModelRowControl::Verifying,
        ModelStatus::Failed { message } => ModelRowControl::Failed(message.clone().into()),
        ModelStatus::Ready if is_selected => ModelRowControl::Selected,
        ModelStatus::Ready if model.deletable() => ModelRowControl::Delete,
        ModelStatus::Ready => ModelRowControl::Provisioned(source_label(model.source)),
    }
}

/// The macOS floor `model_id` declares in the manifest — read from there,
/// never restated as a literal version in this view (ADR-019).
#[must_use]
fn min_macos_label(model_id: &str) -> SharedString {
    vuho_model_paths::manifest()
        .stt
        .model(model_id)
        .map_or_else(
            || SharedString::from("Unsupported on this Mac"),
            |model| SharedString::from(format!("Needs macOS {}", model.min_macos)),
        )
}

/// Who provisioned a model Vuho may not delete. A `None` source is a model
/// that reports `Ready` while the resolver cannot say where it came from —
/// rare, but calling it "Downloaded" next to no Delete button states the one
/// thing that cannot be true of it.
#[must_use]
fn source_label(source: Option<ModelSource>) -> SharedString {
    match source {
        Some(ModelSource::Bundle) => "Bundled".into(),
        Some(ModelSource::DevTree) => "Dev tree".into(),
        Some(ModelSource::EnvOverride) => "Override".into(),
        Some(ModelSource::UserData) => "Downloaded".into(),
        None => "Source unknown".into(),
    }
}

fn download_of(id: String) -> ProvisionCommand {
    ProvisionCommand::Download(id)
}

fn delete_of(id: String) -> ProvisionCommand {
    ProvisionCommand::Delete(id)
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

/// The mic dropdown's current-value label + color for a given [`MicDisplay`]
/// — pure, split out of [`SettingsTab::render_mic_row`] purely to help keep
/// that function short (CONSTITUTION rule 28).
#[must_use]
fn mic_label(display: &MicDisplay) -> (SharedString, Hsla) {
    match display {
        MicDisplay::SystemDefault => ("System Default".into(), theme::TEXT_PRIMARY),
        MicDisplay::Connected(name) => (name.clone().into(), theme::TEXT_PRIMARY),
        MicDisplay::Missing(name) => (name.clone().into(), theme::WARN_AMBER),
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

/// The model combobox's expanded option list. Split out of
/// [`SettingsTab::render_model_row`] (CONSTITUTION rule 28); free-standing
/// because it needs no view state of its own.
fn render_model_options(
    models: &[ModelAvailability],
    cx: &mut Context<SettingsTab>,
) -> impl IntoElement {
    let mut options = controls::dropdown_option_list();
    for (ix, model) in models.iter().enumerate() {
        if !selectable(model) {
            options = options.child(div().px_3().py_2().child(controls::disabled_pill(
                SharedString::from(model.display_name.clone()),
            )));
            continue;
        }
        let id = model.id.clone();
        options = options.child(controls::dropdown_option(
            model.display_name.clone(),
            theme::TEXT_PRIMARY,
            ("settings-tab-model-opt", ix),
            cx.listener(move |view, _event, _window, cx| {
                view.select_model(id.clone(), cx);
            }),
        ));
    }
    options
}

/// A downloading model row's three children: the progress bar, the byte
/// counter, and the in-progress pill.
fn append_download_progress(row: Div, received_bytes: u64, total_bytes: u64) -> Div {
    row.child(theme::progress_bar(
        ModelStatus::Downloading {
            received_bytes,
            total_bytes,
        }
        .fraction()
        .unwrap_or(0.0),
    ))
    .child(model_status_line(
        format!(
            "{} of {}",
            readiness::format_mb(received_bytes),
            readiness::format_mb(total_bytes)
        ),
        theme::TEXT_SECONDARY,
    ))
    .child(controls::disabled_pill("In progress…"))
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
///
/// `permissions_missing` carries each non-granted permission's live
/// [`Access`] (F6) — every row below renders purely from it (falling back to
/// [`Access::Granted`] for a permission not present in the list), never a
/// fresh [`Permission::access`] call at render time, which used to race
/// `panel::start_permissions_poll`'s own 500 ms tick (the collapsed header
/// could disagree with a row for up to that long).
fn render_permissions_section(
    permissions_missing: &[(Permission, Access)],
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
            let access = permission_access(permissions_missing, permission);
            card = card.child(render_permission_row(permission, access, hotkey_failed, cx));
        }
    }

    if launch_blocked && permissions_missing.is_empty() {
        card = card.child(render_relaunch_gate_row(cx));
    }

    card
}

/// `permission`'s [`Access`] as carried by `permissions_missing` (F6), or
/// [`Access::Granted`] when it's absent from that list — every permission
/// not currently missing is, definitionally, granted.
#[must_use]
fn permission_access(
    permissions_missing: &[(Permission, Access)],
    permission: Permission,
) -> Access {
    permissions_missing
        .iter()
        .find_map(|&(p, access)| (p == permission).then_some(access))
        .unwrap_or(Access::Granted)
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
/// note, and an action driven by [`permission_row_kind`]. `access` is the
/// caller's already-derived [`Access`] (F6) — this function never calls
/// [`Permission::access`] itself.
fn render_permission_row(
    permission: Permission,
    access: Access,
    hotkey_failed: bool,
    cx: &mut Context<SettingsTab>,
) -> impl IntoElement {
    let is_accessibility = matches!(permission, Permission::Accessibility);
    let kind = permission_row_kind(access, is_accessibility, hotkey_failed);

    let row = render_permission_row_header(permission, access, is_accessibility);
    append_permission_action(row, permission, kind, cx)
}

/// One permission row's header: label, description, and — for
/// Accessibility specifically, while not yet granted — the relaunch-after-
/// granting note. Split out of [`render_permission_row`] to keep it under
/// the 40-line render-helper limit (CONSTITUTION rule 28).
fn render_permission_row_header(
    permission: Permission,
    access: Access,
    is_accessibility: bool,
) -> Div {
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
        .child(model_status_line(
            permission.description(),
            theme::TEXT_TERTIARY,
        ));

    if is_accessibility && access != Access::Granted {
        row = row.child(model_status_line(
            "Vuho must be relaunched after granting.",
            theme::TEXT_TERTIARY,
        ));
    }
    row
}

/// Append the [`RowKind`]-driven action (✓ Granted / relaunch button /
/// Allow button / Open System Settings button) to a permission row's
/// header. Split out of [`render_permission_row`] (CONSTITUTION rule 28).
fn append_permission_action(
    row: Div,
    permission: Permission,
    kind: RowKind,
    cx: &mut Context<SettingsTab>,
) -> Div {
    match kind {
        RowKind::Granted => row.child(model_status_line("✓ Granted", theme::OK_GREEN)),
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
    }
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

    // ── model_row_control — WP9.S4's row table ──────────────────────────

    fn availability(status: ModelStatus, source: Option<ModelSource>) -> ModelAvailability {
        ModelAvailability {
            id: "some-model".to_owned(),
            display_name: "Some Model".to_owned(),
            status,
            source,
            total_bytes: 636_000_000,
            supported_on_this_os: true,
        }
    }

    #[test]
    fn a_model_this_macos_is_too_old_for_offers_no_action_at_all() {
        let unsupported = ModelAvailability {
            supported_on_this_os: false,
            ..availability(ModelStatus::Missing { total_bytes: 1 }, None)
        };
        assert!(matches!(
            model_row_control(&unsupported, false),
            ModelRowControl::Unsupported(_)
        ));
    }

    #[test]
    fn an_absent_model_offers_a_download() {
        assert_eq!(
            model_row_control(
                &availability(ModelStatus::Missing { total_bytes: 1 }, None),
                false
            ),
            ModelRowControl::Download
        );
    }

    #[test]
    fn a_downloading_model_shows_its_progress() {
        assert_eq!(
            model_row_control(
                &availability(
                    ModelStatus::Downloading {
                        received_bytes: 10,
                        total_bytes: 100
                    },
                    None
                ),
                false
            ),
            ModelRowControl::Downloading {
                received_bytes: 10,
                total_bytes: 100
            }
        );
    }

    #[test]
    fn a_verifying_model_shows_that_it_is_verifying() {
        assert_eq!(
            model_row_control(&availability(ModelStatus::Verifying, None), false),
            ModelRowControl::Verifying
        );
    }

    #[test]
    fn a_failed_model_offers_a_retry_with_its_own_message() {
        assert_eq!(
            model_row_control(
                &availability(
                    ModelStatus::Failed {
                        message: "connection reset".to_owned()
                    },
                    None
                ),
                false
            ),
            ModelRowControl::Failed("connection reset".into())
        );
    }

    #[test]
    fn the_selected_model_is_never_deletable_from_its_own_row() {
        assert_eq!(
            model_row_control(
                &availability(ModelStatus::Ready, Some(ModelSource::UserData)),
                true
            ),
            ModelRowControl::Selected,
            "the selected model must not offer a Delete the provisioning loop would refuse"
        );
    }

    #[test]
    fn an_unselected_downloaded_model_offers_delete() {
        assert_eq!(
            model_row_control(
                &availability(ModelStatus::Ready, Some(ModelSource::UserData)),
                false
            ),
            ModelRowControl::Delete
        );
    }

    /// ADR-020: a bundled, dev-tree, or `VUHO_MODEL_FOLDER` model is not
    /// Vuho's to delete, and the row says who provisioned it rather than
    /// leaving the missing Delete button looking like a bug.
    #[test]
    fn a_model_vuho_did_not_download_says_who_provisioned_it_instead() {
        for (source, label) in [
            (ModelSource::Bundle, "Bundled"),
            (ModelSource::DevTree, "Dev tree"),
            (ModelSource::EnvOverride, "Override"),
        ] {
            assert_eq!(
                model_row_control(&availability(ModelStatus::Ready, Some(source)), false),
                ModelRowControl::Provisioned(label.into())
            );
        }
    }

    /// A `Ready` model whose source the resolver could not name must not
    /// claim Vuho downloaded it — that is exactly the model it refuses to
    /// delete.
    #[test]
    fn a_ready_model_with_an_unresolved_source_does_not_claim_to_be_downloaded() {
        assert_eq!(
            model_row_control(&availability(ModelStatus::Ready, None), false),
            ModelRowControl::Provisioned("Source unknown".into())
        );
    }

    // ── shows_speech_model_section ──────────────────────────────────────

    fn status_model(models: Vec<ModelAvailability>) -> StatusModel {
        StatusModel {
            model: models.first().map(|model| model.status.clone()),
            models,
            engine: EngineState::Loading,
            recording: false,
            hotkey: HotkeyState::Active(HotkeySetting::CapsLock),
            permissions_missing: Vec::new(),
            launch_blocked: false,
            settings_load_warning: None,
        }
    }

    /// ADR-021: the permission-gate startup path never runs the provisioning
    /// thread, so a card rendered here could only show a bare model id, an
    /// empty list, a warmup nobody started, and buttons sending into a
    /// dropped channel.
    #[test]
    fn the_gate_path_shows_no_speech_model_card() {
        let mut status = status_model(Vec::new());
        status.launch_blocked = true;
        assert!(!shows_speech_model_section(&status));
    }

    #[test]
    fn a_reported_model_list_shows_the_speech_model_card() {
        let status = status_model(vec![availability(
            ModelStatus::Ready,
            Some(ModelSource::UserData),
        )]);
        assert!(shows_speech_model_section(&status));
    }

    // ── selectable / model_display_name ─────────────────────────────────

    #[test]
    fn only_a_ready_supported_model_is_selectable() {
        assert!(selectable(&availability(ModelStatus::Ready, None)));
        assert!(!selectable(&availability(
            ModelStatus::Missing { total_bytes: 1 },
            None
        )));
        assert!(!selectable(&ModelAvailability {
            supported_on_this_os: false,
            ..availability(ModelStatus::Ready, None)
        }));
    }

    #[test]
    fn the_combobox_falls_back_to_the_bare_id_before_the_first_model_list_arrives() {
        assert_eq!(model_display_name(&[], "some-model"), "some-model");
        assert_eq!(
            model_display_name(&[availability(ModelStatus::Ready, None)], "some-model"),
            "Some Model"
        );
    }

    // ── reachable_languages ─────────────────────────────────────────────

    /// Both sets come from their owners; this pins that the intersection is
    /// taken rather than one list being silently substituted for the other.
    #[test]
    fn a_canary_backed_model_lists_only_languages_the_os_can_also_report() {
        let mapped = vuho_os_integration::mapped_languages();
        for id in vuho_model_paths::manifest().stt.models.keys() {
            let codes = reachable_languages(id);
            assert!(!codes.is_empty(), "{id} reaches no language at all");
            assert!(
                codes.iter().all(|code| mapped.contains(code)),
                "{id} claims a language the OS layer cannot report"
            );
        }
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
