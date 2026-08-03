//! The Settings window (GPUI): microphone device + hotkey preset dropdowns.
//!
//! Singleton: [`open_settings_window`] re-activates the existing window
//! (via `activate_window`) if one is alive, or opens a fresh one if it was
//! previously closed. Settings apply **live**, save-on-change (no Save
//! button) — matching macOS conventions.
//!
//! Deliberately does *not* call `window_config::apply_window_config` — that
//! module's click-through/level/collection-behavior surgery is
//! overlay-specific (ADR-006); the settings window is a normal, focusable,
//! titled window.

use gpui::{
    div, hsla, prelude::*, px, App, Context, IntoElement, ParentElement, Render, SharedString,
    Size, Styled, TitlebarOptions, Window, WindowBackgroundAppearance, WindowBounds, WindowKind,
    WindowOptions,
};
use vuho_settings::HotkeySetting;

use crate::app_state::VuhoState;
use crate::hotkey_presets::to_hotkey_config;

/// Settings window dimensions.
const SETTINGS_WIDTH: gpui::Pixels = px(360.0);
const SETTINGS_HEIGHT: gpui::Pixels = px(260.0);

/// Open the settings window, or re-activate it if already open.
///
/// Singleton behavior: if `VuhoState::settings_window` holds a handle whose
/// window is still alive, `activate_window()` brings it forward. Otherwise a
/// fresh window is created and the handle stored back into the global. This
/// covers both "never opened" (`None`) and "the user closed it": the fresh
/// window registers an [`on_window_should_close`](Window::on_window_should_close)
/// hook (below) that resets `settings_window` back to `None` at close time,
/// so a closed window is never mistaken for a live one on the next open —
/// belt-and-braces with `handle.update(..)`'s own `Err` on a dead handle,
/// which would otherwise be the only signal.
///
/// The app is `LSUIElement`/accessory (no Dock icon, non-activating), so
/// `cx.activate(true)` is required to actually bring the window forward at
/// the platform level — matching `permissions::activate_app`'s rationale.
pub(crate) fn open_settings_window(cx: &mut App) {
    let existing = cx.global::<VuhoState>().settings_window;
    if let Some(handle) = existing {
        if handle
            .update(cx, |_, window, _| window.activate_window())
            .is_ok()
        {
            cx.activate(true);
            return;
        }
    }

    let size = Size {
        width: SETTINGS_WIDTH,
        height: SETTINGS_HEIGHT,
    };
    let bounds = WindowBounds::centered(size, cx);
    let result = cx.open_window(
        WindowOptions {
            window_bounds: Some(bounds),
            titlebar: Some(TitlebarOptions {
                title: Some(SharedString::from("Vuho Settings")),
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
            window.set_window_title("Vuho Settings");
            // Clear the singleton handle on close — without this, closing
            // and reopening the window would find a stale `Some(handle)`
            // whose window is gone; `handle.update` on it returns `Err`,
            // which the fallthrough above does already tolerate, but
            // clearing it here keeps the global from accumulating a
            // reference to a dead window indefinitely between opens.
            window.on_window_should_close(cx, |_window, cx| {
                cx.global_mut::<VuhoState>().settings_window = None;
                true
            });
            cx.new(SettingsView::new)
        },
    );

    match result {
        Ok(handle) => {
            cx.global_mut::<VuhoState>().settings_window = Some(handle);
            cx.activate(true);
        }
        Err(e) => {
            log::warn!("settings_window: failed to open window: {e}");
        }
    }
}

/// The settings window's root view: a microphone dropdown and a hotkey
/// preset dropdown, each reading/writing the global `SettingsStore`.
pub(crate) struct SettingsView {
    /// Input device names from `vuho_stt_engine::list_input_devices()`,
    /// snapshotted at window-open time. An `Err` (`cpal` host enumeration
    /// failure) or an empty result just means the dropdown offers only
    /// "System Default".
    devices: Vec<String>,
    mic_open: bool,
    hotkey_open: bool,
}

impl SettingsView {
    fn new(_cx: &mut Context<Self>) -> Self {
        let devices = match vuho_stt_engine::list_input_devices() {
            Ok(devices) => devices,
            Err(e) => {
                log::warn!("settings_window: failed to list input devices: {e}");
                Vec::new()
            }
        };
        Self {
            devices,
            mic_open: false,
            hotkey_open: false,
        }
    }

    /// Persist the chosen microphone (`None` = system default) and close
    /// the dropdown. Applied at the *next* session start (ADR-013:
    /// `vuho-audio` resolves the device by name at capture-start time; this
    /// is not live-rebound mid-session the way the hotkey is).
    fn select_microphone(&mut self, choice: Option<String>, cx: &mut Context<Self>) {
        self.mic_open = false;
        let store = cx.global::<VuhoState>().settings.clone();
        if let Err(e) = store.update(|s| s.microphone = choice) {
            log::warn!("settings_window: failed to save microphone setting: {e}");
        }
        cx.notify();
    }

    /// Persist the chosen hotkey preset, close the dropdown, and rebind the
    /// live listener: `stop()` (fast — see the `HotkeyListener::stop` fix)
    /// then `start()` with the new config. A failed `start()` (Accessibility
    /// revoked) prompts the user, deferred via `cx.spawn` for the same
    /// nested-run-loop reason `wire_production` defers its own prompt.
    fn select_hotkey(&mut self, preset: HotkeySetting, cx: &mut Context<Self>) {
        self.hotkey_open = false;

        let state = cx.global::<VuhoState>();
        let store = state.settings.clone();
        let hotkey = state.hotkey.clone();
        let cmd_tx = state.cmd_tx.clone();

        if let Err(e) = store.update(|s| s.hotkey = preset) {
            log::warn!("settings_window: failed to save hotkey setting: {e}");
        }

        let start_result = {
            let mut listener = hotkey.borrow_mut();
            listener.stop();
            listener.start(&cmd_tx, to_hotkey_config(preset))
        };
        if start_result.is_err() {
            cx.spawn(|_this, _cx: &mut gpui::AsyncApp| async move {
                crate::permissions::prompt_accessibility();
            })
            .detach();
        }
        cx.notify();
    }

    /// The microphone row: label + dropdown button, expanding to an option
    /// list ("System Default" + every enumerated device) when open.
    fn render_mic_row(&mut self, current: &str, cx: &mut Context<Self>) -> impl IntoElement {
        let mut column = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(section_label("Microphone"))
            .child(dropdown_button(
                current.to_string(),
                "mic-dropdown",
                cx.listener(|view, _event, _window, cx| {
                    view.mic_open = !view.mic_open;
                    view.hotkey_open = false;
                    cx.notify();
                }),
            ));

        if self.mic_open {
            let mut options = dropdown_option_list();
            options = options.child(dropdown_option(
                "System Default".to_string(),
                ("mic-opt", 0usize),
                cx.listener(|view, _event, _window, cx| {
                    view.select_microphone(None, cx);
                }),
            ));
            for (ix, device) in self.devices.clone().into_iter().enumerate() {
                let device_for_click = device.clone();
                options = options.child(dropdown_option(
                    device,
                    ("mic-opt", ix + 1),
                    cx.listener(move |view, _event, _window, cx| {
                        view.select_microphone(Some(device_for_click.clone()), cx);
                    }),
                ));
            }
            column = column.child(options);
        }
        column
    }

    /// The hotkey row: label + dropdown button, expanding to every
    /// [`HotkeySetting`] preset when open.
    fn render_hotkey_row(
        &mut self,
        current: &'static str,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let mut column = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(section_label("Hotkey"))
            .child(dropdown_button(
                current.to_string(),
                "hotkey-dropdown",
                cx.listener(|view, _event, _window, cx| {
                    view.hotkey_open = !view.hotkey_open;
                    view.mic_open = false;
                    cx.notify();
                }),
            ));

        if self.hotkey_open {
            let mut options = dropdown_option_list();
            for (ix, preset) in HotkeySetting::ALL.into_iter().enumerate() {
                options = options.child(dropdown_option(
                    preset.label().to_string(),
                    ("hotkey-opt", ix),
                    cx.listener(move |view, _event, _window, cx| {
                        view.select_hotkey(preset, cx);
                    }),
                ));
            }
            column = column.child(options);
        }
        column
    }
}

impl Render for SettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let settings = cx.global::<VuhoState>().settings.get();
        let mic_label = settings
            .microphone
            .clone()
            .unwrap_or_else(|| "System Default".to_string());
        let hotkey_label = settings.hotkey.label();

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(hsla(0.7, 0.1, 0.14, 1.0))
            .text_color(hsla(0.0, 0.0, 1.0, 0.95))
            .p_6()
            .gap_6()
            .child(self.render_mic_row(&mic_label, cx))
            .child(self.render_hotkey_row(hotkey_label, cx))
    }
}

/// A small uppercase-ish section label above a dropdown.
fn section_label(text: &'static str) -> impl IntoElement {
    div()
        .text_size(px(12.0))
        .text_color(hsla(0.0, 0.0, 1.0, 0.6))
        .child(text)
}

/// The always-visible dropdown button showing the current value.
fn dropdown_button(
    current: String,
    id: &'static str,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .id(id)
        .flex()
        .justify_between()
        .items_center()
        .px_3()
        .py_2()
        .rounded(px(6.0))
        .bg(hsla(0.0, 0.0, 1.0, 0.08))
        .cursor_pointer()
        .child(current)
        .child("▾")
        .on_click(on_click)
}

/// Container for a dropdown's expanded option list.
fn dropdown_option_list() -> gpui::Div {
    div()
        .flex()
        .flex_col()
        .mt_1()
        .rounded(px(6.0))
        .bg(hsla(0.0, 0.0, 0.0, 0.35))
        .overflow_hidden()
}

/// One selectable row within an expanded dropdown.
fn dropdown_option(
    text: String,
    id: impl Into<gpui::ElementId>,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .id(id.into())
        .px_3()
        .py_2()
        .cursor_pointer()
        .hover(|style| style.bg(hsla(0.0, 0.0, 1.0, 0.1)))
        .child(text)
        .on_click(on_click)
}
