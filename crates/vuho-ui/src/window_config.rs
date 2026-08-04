//! `NSWindow` configuration via raw `msg_send!` surgery.
//!
//! All reach-through to macOS `NSWindow` is isolated here. Every function
//! guards against `None` handles and logs a warning instead of panicking.
//!
//! We use raw `msg_send!` with [`objc2::runtime::AnyObject`] (the `id` type)
//! rather than objc2 wrapper types, because the wrapper crates have
//! granular feature flags that don't cover every selector we need. The two
//! exceptions are [`set_accessory_activation_policy`], which uses the typed
//! `NSApplication::setActivationPolicy` — the raw `objc_msgSend` workaround
//! it used to need is gone (see that function's doc comment history) — and
//! [`set_frame`]'s screen resolution, which uses the typed
//! `NSScreen::screens`/`mainScreen`/`frame` (G2) rather than raw
//! `msg_send!` because the array/`Option` return shapes are awkward to
//! thread through untyped `AnyObject` pointers.
//!
//! **Click/focus model (ARCHITECTURE.md ADR-021):** the panel is always
//! mouse-interactive — there is no click-through mode any more — but must
//! never take keyboard focus, so `inject_text`'s synthesized ⌘V always
//! lands in whatever app the user is dictating into, never the panel
//! itself. [`apply_window_config`] sets `setBecomesKeyOnlyIfNeeded: true`,
//! the `NSPanel` mechanism that accepts clicks on the tab strip, dropdowns,
//! and buttons without ever becoming the key window. No control in the
//! panel takes text input (`SettingsTab` is dropdowns and buttons only), so
//! nothing legitimately needs key status; `main.rs`'s Esc/⌘, keybindings
//! consequently stop firing while the panel isn't key, a known and accepted
//! gap (`TODO.md`).
//!
//! **Deferred surgery (CONSTITUTION rule 33):** `setFrame:display:YES`,
//! `orderFront:`, and `orderOut:` all deliver synchronous `AppKit` delegate
//! callbacks back into gpui (`windowDidMove:`, the content view's
//! `setFrameSize:`) — sending them from inside a live gpui `App` borrow
//! re-enters gpui and corrupts that borrow. `set_frame`, `order_front`, and
//! `order_out` therefore no longer take `&mut Window` at all: they enqueue
//! their `msg_send!` through [`main_queue::defer`], reading the `NSWindow`
//! handle from the module's `WINDOW` `thread_local` inside the deferred
//! body rather than from the caller's `Window` — the same borrow-discipline
//! rule `status_bar.rs`'s module doc documents: clone the handle out of any
//! borrow before sending an `AppKit` message. `apply_window_config`'s own
//! static one-time config (level, collection behavior,
//! `setHidesOnDeactivate:`, `setBecomesKeyOnlyIfNeeded:`) stays inline — it
//! runs once at window creation, on a window that is still hidden and has
//! never been shown, so it triggers no delegate callback gpui could
//! re-enter through.

use std::cell::RefCell;

use gpui::{Bounds, Pixels, Window};
use objc2::ffi::{NSInteger, NSUInteger};
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::NSRect;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use crate::main_queue;

thread_local! {
    /// Main-thread-only storage for the panel's `NSWindow`, retained once at
    /// [`apply_window_config`] time — `main.rs` creates exactly one panel
    /// window per process. Every surgery function below (`set_frame`,
    /// `order_front`, `order_out`) reads this from inside its
    /// `main_queue::defer`red body instead of threading a `&mut Window` all
    /// the way from its own caller, following this module's
    /// borrow-discipline rule (below): clone the handle out of the borrow
    /// before sending any `AppKit` message.
    static WINDOW: RefCell<Option<Retained<AnyObject>>> = const { RefCell::new(None) };
}

/// Window level: above menu bar and fullscreen apps.
///
/// Uses `kCGScreenSaverWindowLevel` (1000) so the overlay floats above
/// the menu bar (100) and any fullscreen content (107).
const OVERLAY_LEVEL: i64 = 1000;

/// Collection behavior bits (matching `NSWindowCollectionBehavior`).
const COLLECTION_CAN_JOIN_ALL_SPACES: NSUInteger = 1 << 8;
const COLLECTION_FULL_SCREEN_AUXILIARY: NSUInteger = 1 << 9;

/// Apply all `NSWindow` overrides required for the panel (ADR-006, ADR-021).
///
/// Called immediately after the GPUI window is created. Sets the
/// click-without-key-status focus model, window level, and collection
/// behavior.
pub(crate) fn apply_window_config(window: &mut Window) {
    let Some(ns_window) = get_ns_window(window) else {
        log::warn!("window_config: could not obtain NSWindow handle");
        return;
    };

    // SAFETY: `ns_window` is a valid, live `NSWindow*` just obtained from
    // `get_ns_window` (a borrowed/unretained pointer per that function's own
    // doc comment) — retaining it here is what lets `WINDOW` hold it safely
    // for the process lifetime, past this function's own call stack.
    let retained = unsafe { Retained::retain(ns_window) };
    WINDOW.with(|w| *w.borrow_mut() = retained);

    // Accept clicks without ever becoming the key window (ADR-021's
    // replacement for click-through) — see the module doc comment's
    // "Click/focus model" section.
    unsafe {
        let _: () = msg_send![ns_window, setBecomesKeyOnlyIfNeeded: true];
    }

    // Above menu bar / fullscreen (ADR-006 criterion 3).
    unsafe {
        #[allow(clippy::cast_possible_truncation)]
        let level: NSInteger = OVERLAY_LEVEL as NSInteger;
        let _: () = msg_send![ns_window, setLevel: level];
    }

    // All-spaces + fullscreen-auxiliary (ADR-006 criterion 3).
    unsafe {
        let behavior: NSUInteger =
            COLLECTION_CAN_JOIN_ALL_SPACES | COLLECTION_FULL_SCREEN_AUXILIARY;
        let _: () = msg_send![ns_window, setCollectionBehavior: behavior];
    }

    // NSPanel defaults hidesOnDeactivate to YES; GPUI never clears it. Left
    // as-is, the panel silently disappears whenever the app activates then
    // deactivates again (e.g. after an NSAlert), independent of our own
    // order_front/order_out calls.
    unsafe {
        let _: () = msg_send![ns_window, setHidesOnDeactivate: false];
    }
}

/// Run `f` on the next main-queue turn ([`main_queue::defer`]) with a
/// reference to the panel's retained `NSWindow` — a no-op (logged) if
/// [`apply_window_config`] hasn't installed [`WINDOW`] yet.
fn defer_on_window(f: impl FnOnce(&AnyObject) + Send + 'static) {
    main_queue::defer(move || {
        WINDOW.with(|w| {
            let borrow = w.borrow();
            let Some(ns_window) = borrow.as_ref() else {
                log::warn!("window_config: NSWindow handle not yet installed");
                return;
            };
            f(ns_window);
        });
    });
}

/// Show the overlay window (objc2 `orderFront:`).
pub(crate) fn order_front() {
    defer_on_window(|ns_window| unsafe {
        let _: () = msg_send![ns_window, orderFront: std::ptr::null_mut::<AnyObject>()];
    });
}

/// Hide the overlay window (objc2 `orderOut:`).
pub(crate) fn order_out() {
    defer_on_window(|ns_window| unsafe {
        let _: () = msg_send![ns_window, orderOut: std::ptr::null_mut::<AnyObject>()];
    });
}

/// Move and resize the window in one atomic `setFrame:display:` call.
///
/// `bounds` is in GPUI's coordinate space: top-left origin, relative to the
/// *primary* display (the same space `Window::bounds()` reads back, and the
/// same space GPUI's window-creation path accepts — vendored gpui
/// mac/window.rs:653-659 for creation, mac/window.rs:515-530 for the
/// matching read-back; every caller of this function derives `bounds` from
/// `cx.primary_display()` — `panel.rs`'s `panel_bounds`). This function converts to Cocoa's bottom-left
/// origin, screen-absolute space using the **primary** screen's frame
/// (`[[NSScreen screens] firstObject]`, falling back to `[NSScreen
/// mainScreen]` only if that array is somehow empty).
///
/// G2: this deliberately never resolves the flip against `[ns_window
/// screen]` (the window's own current screen) — that pointer is `nil`
/// while the window is ordered out (which `panel.rs`'s `hide_root` now
/// triggers a `set_frame` call during, on every dismiss), and even when
/// non-nil it reflects whatever screen the window's *previous* frame
/// happened to overlap, not the screen `bounds` was actually computed
/// against. GPUI's global coordinate space is anchored to the primary
/// display's top-left, and in Cocoa's screen-absolute space the primary
/// screen always sits at origin `(0, 0)` (macOS convention: every other
/// screen's origin is expressed relative to it) — so the primary screen's
/// frame is the one flip reference that is always correct for a
/// primary-anchored `bounds`, regardless of which screen the window
/// currently happens to be on or whether it's on screen at all. Resolving
/// against the wrong screen on a multi-monitor setup silently reinterprets
/// `bounds` in that screen's coordinate space, placing the window offset or
/// off-screen.
pub(crate) fn set_frame(bounds: Bounds<Pixels>) {
    let Some(mtm) = objc2::MainThreadMarker::new() else {
        log::warn!("window_config: set_frame called off the main thread");
        return;
    };
    let screens = objc2_app_kit::NSScreen::screens(mtm);
    let screen = screens
        .firstObject()
        .or_else(|| objc2_app_kit::NSScreen::mainScreen(mtm));
    let Some(screen) = screen else {
        log::warn!("window_config: no screen available to place the window on");
        return;
    };
    let screen_frame = screen.frame();

    let (cocoa_x, cocoa_y) = gpui_origin_to_cocoa(
        bounds.origin.x.to_f64(),
        bounds.origin.y.to_f64(),
        bounds.size.height.to_f64(),
        screen_frame.origin.x,
        screen_frame.origin.y,
        screen_frame.size.height,
    );
    let width = bounds.size.width.to_f64();
    let height = bounds.size.height.to_f64();
    // Only the final `setFrame:display:` send is deferred — the NSScreen
    // lookup and coordinate flip above are pure/read-only (no delegate
    // callback risk) and need the caller's own thread anyway
    // (`MainThreadMarker`). `NSRect` itself doesn't cross the closure
    // boundary (its fields aren't `Send`); its `f64` components do, and the
    // rect is reconstructed from them inside the deferred body.
    defer_on_window(move |ns_window| unsafe {
        let frame = NSRect::new(
            objc2_foundation::NSPoint::new(cocoa_x, cocoa_y),
            objc2_foundation::NSSize::new(width, height),
        );
        let _: () = msg_send![ns_window, setFrame: frame, display: true];
    });
}

/// Pure coordinate flip for [`set_frame`]: GPUI top-left-relative-to-screen
/// origin → Cocoa bottom-left-absolute origin.
///
/// This is the algebraic inverse of gpui's own `bounds()` read-back
/// (vendored gpui mac/window.rs:515-530), so a value written through
/// [`set_frame`] and immediately read back through GPUI's `Window::bounds()`
/// round-trips exactly:
///
/// ```text
/// bounds() computes:  gpui_y = screen_origin_y + screen_height - cocoa_y - height
/// this inverts to:    cocoa_y = screen_origin_y + screen_height - gpui_y - height
/// ```
///
/// The x axis has no flip, only a change of reference frame:
/// `cocoa_x = screen_origin_x + gpui_x`.
fn gpui_origin_to_cocoa(
    gpui_x: f64,
    gpui_y: f64,
    height: f64,
    screen_origin_x: f64,
    screen_origin_y: f64,
    screen_height: f64,
) -> (f64, f64) {
    let cocoa_x = screen_origin_x + gpui_x;
    let cocoa_y = screen_origin_y + screen_height - gpui_y - height;
    (cocoa_x, cocoa_y)
}

/// Set app activation policy to Accessory (no Dock icon, non-activating).
///
/// This prevents the app from appearing in the Dock and from becoming
/// the active (frontmost) app — critical for criterion 1.
///
/// Must be called from the main thread — GPUI's own event loop runs there,
/// so every real caller already satisfies this; if called off the main
/// thread (e.g. a hypothetical future refactor), this logs a warning and is
/// a no-op rather than panicking.
pub(crate) fn set_accessory_activation_policy() {
    let Some(mtm) = objc2::MainThreadMarker::new() else {
        log::warn!("window_config: set_accessory_activation_policy called off the main thread");
        return;
    };
    let ns_app = objc2_app_kit::NSApplication::sharedApplication(mtm);
    ns_app.setActivationPolicy(objc2_app_kit::NSApplicationActivationPolicy::Accessory);
}

/// Get the `NSWindow*` from a GPUI `Window` via `raw_window_handle`.
///
/// The path is:
/// 1. `Window.window_handle()` → `RawWindowHandle::AppKit(handle)`
/// 2. `handle.ns_view` → `*mut c_void` (`NSView` pointer)
/// 3. `[nsView window]` → `NSWindow*`
fn get_ns_window(window: &mut Window) -> Option<*mut AnyObject> {
    // Use the raw-window-handle trait method explicitly, because
    // gpui::Window::window_handle() returns gpui::AnyWindowHandle (opaque).
    let handle = match <Window as HasWindowHandle>::window_handle(window) {
        Ok(h) => h,
        Err(e) => {
            log::warn!("window_config: window_handle error: {e}");
            return None;
        }
    };
    let raw = handle.as_raw();
    let RawWindowHandle::AppKit(appkit) = raw else {
        log::warn!("window_config: unexpected RawWindowHandle variant");
        return None;
    };

    let ns_view_ptr = appkit.ns_view.as_ptr();
    // SAFETY: ns_view_ptr is a valid NSView* owned by GPUI's window, live
    // for the duration of this call. `-[NSView window]` is a plain
    // accessor property (Get rule, not Copy/Create) — it returns an
    // unretained/borrowed NSWindow* owned by AppKit's own window graph, not
    // a +1 reference transferred to us. Every call site below only reads
    // through this raw pointer for one-off msg_send! calls and never
    // releases it, which is the correct handling for a borrowed reference —
    // there's nothing here for us to own or free.
    let ns_window: *mut AnyObject = unsafe {
        let ns_view = ns_view_ptr.cast::<AnyObject>();
        msg_send![ns_view, window]
    };
    if ns_window.is_null() {
        log::warn!("window_config: [nsView window] returned NULL");
        None
    } else {
        Some(ns_window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify constants match macOS documentation.
    #[test]
    fn window_level_constants() {
        // kCGScreenSaverWindowLevel = 1000 (above menu bar at 100, fullscreen at 107).
        assert_eq!(OVERLAY_LEVEL, 1000);
        // set_accessory_activation_policy uses the typed
        // NSApplicationActivationPolicy::Accessory constant directly (value
        // 1, per objc2-app-kit) — no raw literal left to cross-check here.
        assert_eq!(objc2_app_kit::NSApplicationActivationPolicy::Accessory.0, 1);
    }

    /// Reference implementation of gpui's `bounds()` read-back math
    /// (vendored gpui mac/window.rs:515-530), used only to verify the
    /// round trip below — `gpui_origin_to_cocoa` must be its exact inverse.
    fn cocoa_origin_to_gpui(
        cocoa_x: f64,
        cocoa_y: f64,
        height: f64,
        screen_origin_x: f64,
        screen_origin_y: f64,
        screen_height: f64,
    ) -> (f64, f64) {
        let gpui_x = cocoa_x - screen_origin_x;
        let gpui_y = screen_origin_y + screen_height - cocoa_y - height;
        (gpui_x, gpui_y)
    }

    /// Compare two `f64` values computed via different paths of the same
    /// exact (add/subtract-only) arithmetic; `clippy::float_cmp` forbids
    /// `assert_eq!` on floats even when, as here, no rounding is involved.
    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < f64::EPSILON,
            "{actual} vs {expected}"
        );
    }

    /// Primary screen, origin at (0, 0): a window flush with the screen's
    /// top-left should land at the screen's actual top-left in Cocoa's
    /// bottom-left-origin space.
    #[test]
    fn flip_primary_screen_top_left() {
        let (x, y) = gpui_origin_to_cocoa(0.0, 0.0, 200.0, 0.0, 0.0, 1000.0);
        assert_close(x, 0.0);
        assert_close(y, 800.0); // 1000 - 0 - 200
    }

    /// Primary screen: a window flush with the screen's bottom edge should
    /// land at Cocoa y == screen origin y.
    #[test]
    fn flip_primary_screen_bottom_flush() {
        let (_, y) = gpui_origin_to_cocoa(0.0, 800.0, 200.0, 0.0, 0.0, 1000.0);
        assert_close(y, 0.0);
    }

    /// Offset (secondary-screen-like) frame: screen origin away from (0, 0),
    /// including a negative y (a monitor arranged below the primary one).
    #[test]
    fn flip_offset_screen_frame() {
        let (x, y) = gpui_origin_to_cocoa(10.0, 10.0, 300.0, 1920.0, -100.0, 1080.0);
        assert_close(x, 1930.0);
        assert_close(y, -100.0 + 1080.0 - 10.0 - 300.0);
    }

    /// A different screen height (e.g. a 4K display) changes the flip
    /// offset but not the x-axis math.
    #[test]
    fn flip_different_height() {
        let (x, y) = gpui_origin_to_cocoa(50.0, 100.0, 400.0, 0.0, 0.0, 2160.0);
        assert_close(x, 50.0);
        assert_close(y, 2160.0 - 100.0 - 400.0);
    }

    /// Round trip: converting a GPUI-space origin to Cocoa space and back
    /// through gpui's own `bounds()` math must reproduce the original
    /// value exactly, for both a primary and an offset screen.
    #[test]
    fn flip_round_trips_through_bounds_math() {
        let cases: &[(f64, f64, f64, f64, f64, f64)] = &[
            // (gpui_x, gpui_y, height, screen_x, screen_y, screen_height)
            (0.0, 0.0, 200.0, 0.0, 0.0, 1000.0),
            (123.5, 456.25, 300.0, 0.0, 0.0, 1080.0),
            (10.0, 10.0, 300.0, 1920.0, -100.0, 1080.0),
            (0.0, 2000.0, 160.0, -500.0, 200.0, 2160.0),
        ];
        for &(gx, gy, h, sx, sy, sh) in cases {
            let (cx, cy) = gpui_origin_to_cocoa(gx, gy, h, sx, sy, sh);
            let (gx2, gy2) = cocoa_origin_to_gpui(cx, cy, h, sx, sy, sh);
            assert!(
                (gx2 - gx).abs() < f64::EPSILON,
                "x round-trip: {gx2} vs {gx}"
            );
            assert!(
                (gy2 - gy).abs() < f64::EPSILON,
                "y round-trip: {gy2} vs {gy}"
            );
        }
    }
}
