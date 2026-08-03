//! Vuho overlay — always-on-top, semi-transparent, non-focus-stealing panel.
//!
//! Renders a wrapping, bottom-anchored live partial transcript (confirmed
//! solid + unconfirmed dimmed), a breathing red recording-state LED, and a
//! cosmetic activity waveform.
//!
//! Window lifecycle:
//! - Created hidden on app start
//! - Shown on `SessionStarted`
//! - Hidden on `SessionCompleted` / `Error`
//!
//! --demo mode: `cargo run -p vuho-ui --features demo` simulates events
//! with synthetic transcript updates, no mic or engine required.
//!
//! Module map (WP10 split of a former 808+-line `main.rs`): [`event_loop`]
//! owns the poll-and-apply drains + hide/stale-detection logic shared by
//! both production and demo; [`wiring`] owns production-only startup
//! (session, hotkey, menu bar, settings); `demo` (only compiled with
//! `--features demo`, so not a resolvable intra-doc link from a default
//! build) owns the synthetic event generator. `main()` itself only creates
//! the overlay window and picks one of the two.

mod overlay;
mod permissions;
// Settings (global state, hotkey presets, the settings window) and the
// menu-bar status item are production-only wiring (`wiring::wire_production`
// installs them; demo mode has no menu bar or settings) — cfg-gated so they
// aren't dead code under `--features demo`.
#[cfg(not(feature = "demo"))]
mod app_state;
#[cfg(not(feature = "demo"))]
mod app_status;
#[cfg(feature = "demo")]
mod demo;
mod event_loop;
#[cfg(not(feature = "demo"))]
mod hotkey_presets;
#[cfg(not(feature = "demo"))]
mod readiness;
#[cfg(not(feature = "demo"))]
mod settings_window;
#[cfg(not(feature = "demo"))]
mod status_bar;
mod window_config;
#[cfg(not(feature = "demo"))]
mod wiring;

use gpui::{
    prelude::*, px, App, Application, Bounds, Pixels, Point, Size, WindowBounds, WindowKind,
    WindowOptions,
};

// `gpui::actions!` generates unit-struct action markers with no way to
// attach doc comments per-item through the macro — the names (`Quit`,
// `OpenSettings`) are self-explanatory, so this is scoped in its own module
// with a single module-level `allow` (narrower than a crate-wide one) rather
// than threading doc text through a macro that doesn't support it. Not
// `pub(crate)`: a crate-root-level `mod` (private or not) is already
// visible to every module in this binary crate, so `crate::actions::Quit`
// resolves fine from `wiring.rs` without needing to widen this further.
#[allow(missing_docs)]
mod actions {
    gpui::actions!(vuho, [Quit, OpenSettings]);
}
use actions::Quit;

/// Distance from the bottom of the display to the overlay's bottom edge.
const OVERLAY_BOTTOM_MARGIN: Pixels = px(120.0);

/// Overlay window dimensions. Grown (Fix 3, from 420x140) to fit 3 wrapped
/// transcript lines + waveform + status row without crowding the rounded
/// panel edges.
const OVERLAY_WIDTH: Pixels = px(460.0);
const OVERLAY_HEIGHT: Pixels = px(180.0);

/// Compute the overlay's top-left origin so it sits horizontally centered and
/// `bottom_margin` above the bottom edge of the given display bounds.
///
/// Pure helper (unit-tested); `overlay_bounds` handles the display lookup.
fn bottom_center_origin(
    display: Bounds<Pixels>,
    size: Size<Pixels>,
    bottom_margin: Pixels,
) -> Point<Pixels> {
    Point {
        x: display.origin.x + (display.size.width - size.width) * 0.5,
        y: display.origin.y + display.size.height - size.height - bottom_margin,
    }
}

/// Bottom-center window bounds on the primary display; falls back to
/// screen-centered when no display is available.
fn overlay_bounds(size: Size<Pixels>, cx: &App) -> WindowBounds {
    if let Some(display) = cx.primary_display() {
        let origin = bottom_center_origin(display.bounds(), size, OVERLAY_BOTTOM_MARGIN);
        return WindowBounds::Windowed(Bounds { origin, size });
    }
    WindowBounds::centered(size, cx)
}

fn main() {
    // Default to `info` — env_logger's own default is `error`, which silently
    // discards every `info!` in the pipeline. `RUST_LOG` still overrides.
    if let Err(e) =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init()
    {
        log::warn!("vuho: failed to init logger: {e}");
    }
    Application::new().run(move |cx: &mut App| {
        bind_quit_hotkey(cx);

        // Set accessory activation policy: no Dock icon, non-activating.
        window_config::set_accessory_activation_policy();

        if run_preflight_gate_and_check_if_blocked(cx) {
            return;
        }

        let overlay = create_overlay_window(cx);

        #[cfg(feature = "demo")]
        demo::run_demo_mode(overlay, cx);

        #[cfg(not(feature = "demo"))]
        wiring::wire_production(overlay, cx);
    });
}

/// Bind the Cmd+Option+Shift+Q quit action.
///
/// `LSUIElement=true` → no Dock icon, so Cmd+Q is unavailable. This is
/// **not** the primary quit path, despite older documentation here having
/// claimed so: `cx.on_action` dispatches through GPUI's own key-window
/// responder chain, and this app is an accessory app whose overlay window
/// is created with `focus: false` and stays non-key for its entire
/// lifetime — so this binding can only ever fire while some *other* GPUI
/// window (the settings window, or the ADR-016 permission gate window)
/// happens to be the key window. The reliable, always-available quit path
/// — reachable with no window focused at all, which is the app's normal
/// steady state — is the status-bar menu's "Quit Vuho" item
/// (`status_bar.rs`'s `quit:` action, wired in both `install` and
/// `install_gate`). This binding exists purely as a keyboard-only
/// convenience for whenever a window *does* happen to be focused.
fn bind_quit_hotkey(cx: &mut App) {
    cx.bind_keys([gpui::KeyBinding::new("cmd-alt-shift-q", Quit, None)]);
    cx.on_action(|_: &Quit, _cx: &mut App| {
        std::process::exit(0);
    });
}

/// Preflight permission gate (ADR-016): check every required TCC grant
/// *before* any real work (model warmup, hotkey start). If anything is
/// missing, show one gate window and return `true` — the caller must stop
/// there: no overlay, no model warmup, no hotkey, until the user grants
/// everything and relaunches. Returns `false` (a no-op) when nothing is
/// missing, falling straight through to the unchanged path. Always `false`
/// under `--features demo` (no permissions to gate in the demo).
fn run_preflight_gate_and_check_if_blocked(cx: &mut App) -> bool {
    #[cfg(feature = "demo")]
    {
        let _ = cx;
        false
    }
    #[cfg(not(feature = "demo"))]
    {
        let missing = readiness::missing_permissions();
        if missing.is_empty() {
            return false;
        }
        // Install the menu-bar icon *before* opening the gate window:
        // without it, a permission-blocked launch has no menu-bar
        // affordance at all (Fix 2) — the user has no way to tell Vuho is
        // even running, let alone re-front the gate if it gets buried again.
        //
        // `gate_tx`/`gate_rx` bridge the status-bar delegate's
        // "Permissions…" action (an AppKit callback with no `App` access)
        // into `spawn_gate_command_drain`, which can reopen the gate window
        // if the user closed it (Fix 5).
        let (gate_tx, gate_rx) = crossbeam_channel::unbounded();
        status_bar::install_gate(gate_tx);
        readiness::open_permission_gate_window(missing, cx);
        readiness::spawn_gate_command_drain(gate_rx, cx);
        true
    }
}

/// Create the overlay window (hidden initially), anchored bottom-center.
fn create_overlay_window(cx: &mut App) -> gpui::WindowHandle<overlay::OverlayModel> {
    let size = Size {
        width: OVERLAY_WIDTH,
        height: OVERLAY_HEIGHT,
    };
    let bounds = overlay_bounds(size, cx);
    cx.open_window(
        WindowOptions {
            window_bounds: Some(bounds),
            titlebar: None,
            focus: false,
            show: false, // hidden until SessionStarted
            kind: WindowKind::PopUp,
            is_resizable: false,
            is_minimizable: false,
            window_background: gpui::WindowBackgroundAppearance::Transparent,
            ..Default::default()
        },
        |window, cx| {
            window.set_window_title("Vuho Overlay");
            window_config::apply_window_config(window);
            cx.new(|cx| overlay::OverlayModel::new(window, cx))
        },
    )
    .expect("failed to create overlay window")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottom_center_origin_centers_and_anchors_bottom() {
        let display = Bounds {
            origin: Point {
                x: px(0.0),
                y: px(0.0),
            },
            size: Size {
                width: px(1000.0),
                height: px(800.0),
            },
        };
        let size = Size {
            width: OVERLAY_WIDTH,
            height: OVERLAY_HEIGHT,
        };
        let origin = bottom_center_origin(display, size, px(120.0));
        // Horizontally centered: (1000 - 460) / 2 = 270.
        assert_eq!(origin.x, px(270.0));
        // Bottom-anchored: 800 - 180 - 120 = 500.
        assert_eq!(origin.y, px(500.0));
    }

    #[test]
    fn bottom_center_origin_respects_display_offset() {
        // A secondary display offset to the right and down.
        let display = Bounds {
            origin: Point {
                x: px(1000.0),
                y: px(100.0),
            },
            size: Size {
                width: px(800.0),
                height: px(600.0),
            },
        };
        let size = Size {
            width: px(400.0),
            height: px(100.0),
        };
        let origin = bottom_center_origin(display, size, px(50.0));
        // x: 1000 + (800 - 400) / 2 = 1200.
        assert_eq!(origin.x, px(1200.0));
        // y: 100 + 600 - 100 - 50 = 550.
        assert_eq!(origin.y, px(550.0));
    }
}
