//! `NSWindow` configuration via raw `msg_send!` surgery.
//!
//! All reach-through to macOS `NSWindow` is isolated here. Every function
//! guards against `None` handles and logs a warning instead of panicking.
//!
//! We use raw `msg_send!` with [`objc2::runtime::AnyObject`] (the `id` type)
//! rather than objc2 wrapper types, because the wrapper crates have
//! granular feature flags that don't cover every selector we need. The one
//! exception is [`set_accessory_activation_policy`], which uses the typed
//! `NSApplication::setActivationPolicy` — the raw `objc_msgSend` workaround
//! it used to need is gone (see that function's doc comment history).

use gpui::{Bounds, Pixels, Window};
use objc2::ffi::{NSInteger, NSUInteger};
use objc2::runtime::AnyObject;
use objc2::{class, msg_send};
use objc2_foundation::NSRect;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

/// Window level: above menu bar and fullscreen apps.
///
/// Uses `kCGScreenSaverWindowLevel` (1000) so the overlay floats above
/// the menu bar (100) and any fullscreen content (107).
const OVERLAY_LEVEL: i64 = 1000;

/// Collection behavior bits (matching `NSWindowCollectionBehavior`).
const COLLECTION_CAN_JOIN_ALL_SPACES: NSUInteger = 1 << 8;
const COLLECTION_FULL_SCREEN_AUXILIARY: NSUInteger = 1 << 9;

/// Apply all `NSWindow` overrides required for the overlay (ADR-006).
///
/// Called immediately after the GPUI window is created.
/// Sets click-through, window level, and collection behavior.
pub(crate) fn apply_window_config(window: &mut Window) {
    let Some(ns_window) = get_ns_window(window) else {
        log::warn!("window_config: could not obtain NSWindow handle");
        return;
    };

    // Click-through (ADR-006 criterion 2).
    unsafe {
        let _: () = msg_send![ns_window, setIgnoresMouseEvents: true];
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

/// Show the overlay window (objc2 `orderFront:`).
pub(crate) fn order_front(window: &mut Window) {
    let Some(ns_window) = get_ns_window(window) else {
        return;
    };
    unsafe {
        let _: () = msg_send![ns_window, orderFront: std::ptr::null_mut::<AnyObject>()];
    }
}

/// Hide the overlay window (objc2 `orderOut:`).
pub(crate) fn order_out(window: &mut Window) {
    let Some(ns_window) = get_ns_window(window) else {
        return;
    };
    unsafe {
        let _: () = msg_send![ns_window, orderOut: std::ptr::null_mut::<AnyObject>()];
    }
}

/// Toggle click-through (objc2 `setIgnoresMouseEvents:`).
///
/// `apply_window_config` sets this to `true` unconditionally at window
/// creation for the panel; `panel.rs`'s presentation-surgery chokepoint
/// flips it off for the Full presentation (which accepts clicks, e.g. the
/// tab strip and Settings controls) and back on for the Hud presentation
/// (click-through, like the old overlay).
pub(crate) fn set_click_through(window: &mut Window, on: bool) {
    let Some(ns_window) = get_ns_window(window) else {
        return;
    };
    unsafe {
        let _: () = msg_send![ns_window, setIgnoresMouseEvents: on];
    }
}

/// Synchronously make the window key and order it to the front (objc2
/// `makeKeyAndOrderFront:`, nil sender).
///
/// Exists because GPUI's own `Window::activate_window()` is
/// executor-deferred and races with synchronous `order_out` calls made in
/// the same tick. Must never be paired with app activation: the app runs as
/// an accessory (no Dock icon, never frontmost) and the panel itself is
/// non-activating (`NSWindowStyleMaskNonactivatingPanel`, set by GPUI for
/// `WindowKind::PopUp`) — `makeKeyAndOrderFront:` gives it key status
/// without activating the application.
///
/// `panel.rs`'s presentation-surgery chokepoint calls this for the Full
/// presentation only — the Hud presentation stays non-key (click-through,
/// no keyboard focus stolen from whatever app the user is dictating into).
pub(crate) fn make_key_and_order_front(window: &mut Window) {
    let Some(ns_window) = get_ns_window(window) else {
        return;
    };
    unsafe {
        let _: () = msg_send![ns_window, makeKeyAndOrderFront: std::ptr::null_mut::<AnyObject>()];
    }
}

/// Move and resize the window in one atomic `setFrame:display:` call.
///
/// `bounds` is in GPUI's coordinate space: top-left origin, relative to the
/// window's own screen (the same space `Window::bounds()` reads back, and
/// the same space GPUI's window-creation path accepts — vendored gpui
/// mac/window.rs:653-659 for creation, mac/window.rs:515-530 for the
/// matching read-back). This function converts to Cocoa's bottom-left
/// origin, screen-absolute space using the window's own current screen
/// (`[[ns_window screen] frame]`, falling back to `[NSScreen mainScreen]`
/// if the window isn't currently on any screen), so a `bounds()` call
/// immediately after this one reads back the same value that was set.
///
/// `panel.rs`'s presentation-surgery chokepoint is the one caller: it
/// re-frames the panel between the Hud's bottom-center bounds and the Full
/// presentation's centered bounds on every presentation change.
pub(crate) fn set_frame(window: &mut Window, bounds: Bounds<Pixels>) {
    let Some(ns_window) = get_ns_window(window) else {
        return;
    };
    unsafe {
        let mut screen: *mut AnyObject = msg_send![ns_window, screen];
        if screen.is_null() {
            screen = msg_send![class!(NSScreen), mainScreen];
        }
        if screen.is_null() {
            log::warn!("window_config: no screen available to place the window on");
            return;
        }
        let screen_frame: NSRect = msg_send![screen, frame];

        let (cocoa_x, cocoa_y) = gpui_origin_to_cocoa(
            bounds.origin.x.to_f64(),
            bounds.origin.y.to_f64(),
            bounds.size.height.to_f64(),
            screen_frame.origin.x,
            screen_frame.origin.y,
            screen_frame.size.height,
        );
        let frame = NSRect::new(
            objc2_foundation::NSPoint::new(cocoa_x, cocoa_y),
            objc2_foundation::NSSize::new(bounds.size.width.to_f64(), bounds.size.height.to_f64()),
        );
        let _: () = msg_send![ns_window, setFrame: frame, display: true];
    }
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
