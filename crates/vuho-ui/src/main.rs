//! Vuho: a single, always-on-top, non-activating panel (`panel::PanelRoot`)
//! with two presentations — Hud (the dictation overlay: bottom-center,
//! click-through, semi-transparent) and Full (a centered, opaque, tabbed
//! Overlay/Settings window). See ARCHITECTURE.md ADR-021 for the design —
//! this replaces the former three-window design (a hidden overlay popup, a
//! lazily-opened settings window, and a separate permission/model readiness
//! window).
//!
//! --demo mode: `cargo run -p vuho-ui --features demo` simulates dictation
//! events with synthetic transcript updates, no mic or engine required —
//! the panel never leaves the Hud presentation in demo mode (no
//! `StatusModel`/`SettingsTab`/permissions to show a Full presentation of).
//!
//! Module map (WP10 split of a former 808+-line `main.rs`, WP6 rehaul):
//! [`panel`] owns the window itself and its two presentations; [`event_loop`]
//! owns the poll-and-apply drains + hide/stale-detection logic shared by
//! both production and demo; [`wiring`] owns production-only startup
//! (dictation session, hotkey, provisioning, status-bar item); `demo` (only
//! compiled with `--features demo`) owns the synthetic event generator.
//! `main()` itself creates the panel and picks one of the two paths.

mod overlay;
mod panel;
mod permissions;
// Settings (global state, hotkey presets) and the menu-bar status item are
// production-only wiring (`wiring::wire_production` installs them; demo
// mode has no menu bar or settings) — cfg-gated so they aren't dead code
// under `--features demo`.
#[cfg(not(feature = "demo"))]
mod app_state;
#[cfg(not(feature = "demo"))]
mod app_status;
// The `Assets` `AssetSource` (icons/*.svg) is always compiled — the panel's
// tab strip needs the icons in both builds (demo renders the Hud only, but
// still creates the same GPUI `Application::new().with_assets(..)`).
mod assets;
#[cfg(not(feature = "demo"))]
mod controls;
#[cfg(feature = "demo")]
mod demo;
mod event_loop;
#[cfg(not(feature = "demo"))]
mod hotkey_presets;
#[cfg(not(feature = "demo"))]
mod readiness;
#[cfg(not(feature = "demo"))]
mod settings_tab;
#[cfg(not(feature = "demo"))]
mod status_bar;
mod theme;
mod window_config;
#[cfg(not(feature = "demo"))]
mod wiring;

use gpui::{App, Application};
#[cfg(not(feature = "demo"))]
use gpui::{AppContext as _, Entity};

// `gpui::actions!` generates unit-struct action markers with no way to
// attach doc comments per-item through the macro — the names (`Quit`,
// `SelectSettingsTab`, `ClosePanel`) are self-explanatory, so this is
// scoped in its own module with a single module-level `allow` (narrower
// than a crate-wide one) rather than threading doc text through a macro
// that doesn't support it.
// Not `pub(crate)`: a crate-root-level `mod` (private or not) is already
// visible to every module in this binary crate, so `crate::actions::Quit`
// resolves fine from `wiring.rs` without needing to widen this further.
#[allow(missing_docs)]
mod actions {
    gpui::actions!(vuho, [Quit, SelectSettingsTab, ClosePanel]);
}
use actions::Quit;

fn main() {
    // Default to `info` — env_logger's own default is `error`, which silently
    // discards every `info!` in the pipeline. `RUST_LOG` still overrides.
    if let Err(e) =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init()
    {
        log::warn!("vuho: failed to init logger: {e}");
    }
    Application::new().with_assets(assets::Assets).run(move |cx: &mut App| {
        bind_quit_hotkey(cx);

        // Set accessory activation policy: no Dock icon, non-activating.
        window_config::set_accessory_activation_policy();

        #[cfg(feature = "demo")]
        {
            let panel = panel::create_panel(cx);
            demo::run_demo_mode(panel, cx);
        }

        #[cfg(not(feature = "demo"))]
        run_production(cx);
    });
}

/// Bind the Cmd+Option+Shift+Q quit action.
///
/// `LSUIElement=true` → no Dock icon, so Cmd+Q is unavailable. This is
/// **not** the primary quit path, despite older documentation here having
/// claimed so: `cx.on_action` dispatches through GPUI's own key-window
/// responder chain, and this app is an accessory app whose panel is created
/// with `focus: false` and stays non-key while presenting the Hud — so this
/// binding can only ever fire while the panel happens to be key (the Full
/// presentation, which does take key status) or some other GPUI window is.
/// The reliable, always-available quit path — reachable with no window
/// focused at all, which is the app's normal steady state — is the
/// status-bar menu's "Quit Vuho" item (`status_bar.rs`'s `quit:` action).
/// This binding exists purely as a keyboard-only convenience for whenever a
/// window *does* happen to be focused.
fn bind_quit_hotkey(cx: &mut App) {
    cx.bind_keys([gpui::KeyBinding::new("cmd-alt-shift-q", Quit, None)]);
    cx.on_action(|_: &Quit, _cx: &mut App| {
        std::process::exit(0);
    });
}

/// The production entry path (`--features demo` never reaches this — see
/// `main`). Builds everything that must exist *before* deciding whether
/// startup is blocked on a missing permission: the settings store, the
/// `StatusModel`/`SettingsTab` entities (shared by the panel and, once
/// wiring runs, the dictation session), and the panel itself. See
/// ARCHITECTURE.md ADR-021 — this is the merged preflight-gate-into-panel
/// design that replaces the old `run_preflight_gate_and_check_if_blocked`.
///
/// Kept short (CONSTITUTION rule 28) by delegating the `StatusModel` build
/// to [`build_status_model`] and the permissions/relaunch-blocked path to
/// [`run_gate_blocked`].
#[cfg(not(feature = "demo"))]
fn run_production(cx: &mut App) {
    use std::sync::Arc;

    use crossbeam_channel::unbounded;

    // Single owner of the settings file for the process (CONSTITUTION rule 1).
    let settings = Arc::new(vuho_settings::SettingsStore::load_or_default());

    let (ui_tx, ui_rx) = unbounded::<app_state::UiCommand>();
    let (provision_tx, provision_rx) = unbounded::<wiring::ProvisionCommand>();

    let status = build_status_model(&settings, cx);
    let settings_tab = cx.new(|cx| {
        settings_tab::SettingsTab::new(status.clone(), settings.clone(), provision_tx, cx)
    });
    let panel = panel::create_panel(status.clone(), settings_tab.clone(), cx);

    let missing = readiness::missing_permissions();
    if !missing.is_empty() {
        run_gate_blocked(cx, panel, &status, missing, ui_tx, ui_rx);
        return;
    }

    wiring::wire_production(
        panel,
        &settings,
        ui_tx,
        ui_rx,
        provision_rx,
        status,
        &settings_tab,
        cx,
    );
    bind_select_settings_tab(cx, panel);
    bind_close_panel(cx, panel);
}

/// Build the shared `StatusModel` entity and wire the tray-sync observer.
/// Split out of [`run_production`] (CONSTITUTION rule 28).
///
/// `initial_hotkey`/`settings_load_warning` are read from `settings` up
/// front, not inside the `cx.new` builder closure below: `cx.new` requires
/// its builder closure to be `'static`, which a closure borrowing `settings`
/// (still needed by value further down in `run_production`) cannot satisfy.
///
/// Initial `hotkey` is `Active(preset)`, corrected to `Failed` by
/// `wiring::start_hotkey` if the listener actually fails to start — a
/// transient over-optimistic read for however long that takes. Never
/// reached on the permissions-blocked path (`wire_production` never runs
/// there), so that transience is production-only.
#[cfg(not(feature = "demo"))]
fn build_status_model(
    settings: &vuho_settings::SettingsStore,
    cx: &mut App,
) -> Entity<app_status::StatusModel> {
    use app_status::{EngineState, HotkeyState, StatusModel};

    let initial_hotkey = settings.get().hotkey;
    // `SharedString::from(&str)` requires the borrow to be `'static`
    // (`SharedString` wraps `ArcCow<'static, str>`) — `load_warning()`'s
    // `&str` is borrowed from `settings` itself, so it must be turned into
    // an owned `String` first.
    let settings_load_warning = settings
        .load_warning()
        .map(|warning| gpui::SharedString::from(warning.to_owned()));

    let status = cx.new(|_| StatusModel {
        model: None,
        engine: EngineState::Loading,
        recording: false,
        hotkey: HotkeyState::Active(initial_hotkey),
        permissions_missing: Vec::new(),
        launch_blocked: false,
        settings_load_warning,
    });
    // Keep the tray in sync with every future `StatusModel` change — cheap
    // no-op until `status_bar::install` actually runs (`run_gate_blocked` or
    // `wiring::wire_production`, whichever branch `run_production` takes).
    cx.observe(&status, |status, cx| {
        status_bar::sync(&status.read(cx).composite());
    })
    .detach();
    status
}

/// The permissions/relaunch-blocked startup path (ADR-021, amending
/// ADR-016): no dictation session is possible yet, so `wiring::wire_production`
/// never runs this launch — the user must grant everything and relaunch.
/// The panel opened on its Settings tab *is* the gate: permission rows live
/// there, and once every grant lands the composite status becomes
/// `RelaunchRequired`, whose relaunch row (also in the Settings tab) is the
/// way forward. Split out of [`run_production`] (CONSTITUTION rule 28).
#[cfg(not(feature = "demo"))]
fn run_gate_blocked(
    cx: &mut App,
    panel: gpui::WindowHandle<panel::PanelRoot>,
    status: &Entity<app_status::StatusModel>,
    missing: Vec<(readiness::Permission, readiness::Access)>,
    ui_tx: crossbeam_channel::Sender<app_state::UiCommand>,
    ui_rx: crossbeam_channel::Receiver<app_state::UiCommand>,
) {
    status.update(cx, |model, cx| {
        model.launch_blocked = true;
        // Seeded with the same derivation (`readiness::missing_permissions()`)
        // `panel::start_permissions_poll` writes on every tick — see
        // `StatusModel::permissions_missing`'s doc comment for why this one
        // synchronous write coexists with the poll instead of leaving the
        // field to the poll alone (F4): without it, the tray's very first
        // paint (below, via `status_bar::install`) would see an empty
        // `permissions_missing` and derive `RelaunchRequired` instead of
        // `PermissionsMissing` for one visible frame, since the poll's own
        // (now-immediate) first tick hasn't run yet at this point — it's
        // spawned by `panel::show_full` further down, and `cx.spawn`'d tasks
        // don't run synchronously inline.
        model.permissions_missing = missing;
        cx.notify();
    });
    status_bar::install(None, ui_tx, &status.read(cx).composite());
    panel::show_full(panel, panel::Tab::Settings, cx);
    event_loop::spawn_ui_drain(panel, ui_rx, status.clone(), cx);
    bind_select_settings_tab(cx, panel);
    bind_close_panel(cx, panel);
}

/// Bind Cmd+, to open the panel on its Settings tab — the reliable,
/// GPUI-native backup to the status-bar menu's "Open Vuho" (which lands on
/// whichever tab was last active, not necessarily Settings). Bound on both
/// the permissions-blocked and production paths, so it works from the very
/// first launch.
#[cfg(not(feature = "demo"))]
fn bind_select_settings_tab(cx: &mut App, panel: gpui::WindowHandle<panel::PanelRoot>) {
    cx.bind_keys([gpui::KeyBinding::new(
        "cmd-,",
        actions::SelectSettingsTab,
        None,
    )]);
    cx.on_action(move |_action: &actions::SelectSettingsTab, cx: &mut App| {
        panel::show_full(panel, panel::Tab::Settings, cx);
    });
}

/// Bind Esc to close the Full presentation (`panel::hide`, F1) — the
/// keyboard-reachable dismiss path alongside the tab strip's own "✕"
/// button. Bound on both the permissions-blocked and production paths
/// (mirrors [`bind_select_settings_tab`] immediately above), so it works
/// from the very first launch.
///
/// Bindings only fire while a GPUI window is key; the panel's Full
/// presentation is the only window in this app that ever takes key status
/// (`window_config::make_key_and_order_front`'s doc comment — the Hud stays
/// non-key/click-through), so this can never intercept Esc from some other
/// context.
#[cfg(not(feature = "demo"))]
fn bind_close_panel(cx: &mut App, panel: gpui::WindowHandle<panel::PanelRoot>) {
    cx.bind_keys([gpui::KeyBinding::new("escape", actions::ClosePanel, None)]);
    cx.on_action(move |_action: &actions::ClosePanel, cx: &mut App| {
        panel::hide(panel, cx);
    });
}
