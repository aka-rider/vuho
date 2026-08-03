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

use gpui::Window;
use objc2::ffi::{NSInteger, NSUInteger};
use objc2::msg_send;
use objc2::runtime::AnyObject;
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
}

/// Show the overlay window (objc2 `orderFront:`).
pub(crate) fn show_overlay(window: &mut Window) {
    let Some(ns_window) = get_ns_window(window) else {
        return;
    };
    unsafe {
        let _: () = msg_send![ns_window, orderFront: std::ptr::null_mut::<AnyObject>()];
    }
}

/// Hide the overlay window (objc2 `orderOut:`).
pub(crate) fn hide_overlay(window: &mut Window) {
    let Some(ns_window) = get_ns_window(window) else {
        return;
    };
    unsafe {
        let _: () = msg_send![ns_window, orderOut: std::ptr::null_mut::<AnyObject>()];
    }
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
}
