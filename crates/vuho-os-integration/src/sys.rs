//! Raw FFI declarations for Carbon, Security, `ApplicationServices`, and
//! CoreGraphics frameworks.
//!
//! Each `extern` block is grouped by framework with an explicit `#[link]` attribute.
//! Safe wrappers live in the same module and handle reference ownership.

use std::os::raw::c_void;

// ── Carbon.framework — TIS (Text Input Services) ──────────────────────────

#[link(name = "Carbon", kind = "framework")]
extern "C" {
    /// Returns a retained `TISInputSourceRef` (`CFTypeRef` subclass).
    ///
    /// **Ownership:** Copy rule — caller owns a +1 reference.
    /// objc2's `CFRetained` handles the release via Drop.
    pub(crate) fn TISCopyCurrentKeyboardInputSource() -> *mut c_void;

    /// Returns a `CFTypeRef` (typically `CFArrayRef` of `CFStringRef`) for the
    /// given property of `source`.
    ///
    /// **Ownership:** Get rule — the returned value is borrowed from `source`
    /// (not retained on the caller's behalf) and must not be released by the
    /// caller without first taking an explicit `CFRetain` of its own; it is
    /// only valid for as long as `source` itself is alive.
    pub(crate) fn TISGetInputSourceProperty(
        source: *mut c_void,
        propertyKey: *mut c_void,
    ) -> *mut c_void;

    /// `kTISPropertyInputSourceLanguages` constant — do NOT release.
    pub(crate) static kTISPropertyInputSourceLanguages: *mut c_void;
}

// ── Security.framework ────────────────────────────────────────────────────

#[link(name = "Security", kind = "framework")]
extern "C" {
    /// Returns `1` if secure event input (e.g. password field focused) is active.
    pub(crate) fn IsSecureEventInputEnabled() -> std::os::raw::c_int;
}

// ── ApplicationServices.framework ─────────────────────────────────────────

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    /// Returns `true` if the current process has accessibility access.
    pub(crate) fn AXIsProcessTrusted() -> bool;

    /// Prompting variant: when `options` contains
    /// `kAXTrustedCheckOptionPrompt: true`, macOS shows the system Accessibility
    /// dialog and adds this process to the Accessibility list.
    pub(crate) fn AXIsProcessTrustedWithOptions(options: *const c_void) -> bool;

    /// `CFStringRef` option key for [`AXIsProcessTrustedWithOptions`].
    ///
    /// objc2 does not provide this constant, so it is declared here (R5). The
    /// symbol's value is a `CFStringRef` (a `const struct __CFString *`).
    pub(crate) static kAXTrustedCheckOptionPrompt: *const c_void;
}

// ── CoreGraphics.framework — Event tap ────────────────────────────────────

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    /// Creates a new event tap. Returns a retained `CFMachPortRef` on success.
    pub(crate) fn CGEventTapCreate(
        tap: objc2_core_graphics::CGEventTapLocation,
        place: objc2_core_graphics::CGEventTapPlacement,
        options: objc2_core_graphics::CGEventTapOptions,
        events_of_interest: objc2_core_graphics::CGEventMask,
        callback: objc2_core_graphics::CGEventTapCallBack,
        user_info: *mut c_void,
    ) -> *mut c_void;
}

// ── CoreFoundation.framework — run loop ────────────────────────────────────
//
// `CFRunLoopRunInMode` is a CoreFoundation symbol, not CoreGraphics — it
// used to be declared inside the CoreGraphics `extern "C"` block above,
// which happened to link successfully only because CoreGraphics itself
// transitively links CoreFoundation on macOS. Declared under its own
// honestly-labeled `#[link]` here instead of relying on that transitive
// link accidentally covering it — this framework is what the symbol
// actually belongs to, and is what objc2-core-foundation's own
// `CFRunLoop`/`CFMachPort` types (used alongside this call in `hotkey.rs`)
// are bound against too.

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    /// Runs the run loop for the given mode and duration.
    pub(crate) fn CFRunLoopRunInMode(
        mode: *mut c_void,
        seconds: f64,
        return_after_source_handled: bool,
    ) -> i32;
}

// ── IOKit.framework — HID access checks (Input Monitoring) ────────────────

#[link(name = "IOKit", kind = "framework")]
extern "C" {
    /// Checks whether the process has access to the given `IOHIDRequestType`.
    /// A pure query — unlike `IOHIDRequestAccess`, this never prompts.
    ///
    /// Returns an `IOHIDAccessType` value (`kIOHIDAccessTypeGranted = 0`,
    /// `kIOHIDAccessTypeDenied = 1`, `kIOHIDAccessTypeUnknown = 2`), declared
    /// `u32` here since `<IOKit/hidsystem/IOHIDLib.h>` defines it as a plain
    /// (non-negative-only) C enum.
    fn IOHIDCheckAccess(request_type: u32) -> u32;

    /// Requests Input Monitoring access from the user for the given
    /// `IOHIDRequestType`, triggering the native system prompt on first call.
    fn IOHIDRequestAccess(request_type: u32) -> bool;
}

/// `kIOHIDRequestTypeListenEvent` from `<IOKit/hidsystem/IOHIDLib.h>` — the
/// request type covering `IOHIDManager`/`IOHIDDevice`/`CGEventTap` listen
/// access ("Input Monitoring" in System Settings). Confirmed against the
/// macOS SDK header: `typedef enum { kIOHIDRequestTypePostEvent,
/// kIOHIDRequestTypeListenEvent } IOHIDRequestType;`.
const K_IOHID_REQUEST_TYPE_LISTEN_EVENT: u32 = 1;

/// `kIOHIDAccessTypeGranted` from the same header:
/// `typedef enum { kIOHIDAccessTypeGranted, kIOHIDAccessTypeDenied,
/// kIOHIDAccessTypeUnknown } IOHIDAccessType;`.
const K_IOHID_ACCESS_TYPE_GRANTED: u32 = 0;

/// `kIOHIDAccessTypeDenied` from the same header.
const K_IOHID_ACCESS_TYPE_DENIED: u32 = 1;

/// `kIOHIDAccessTypeUnknown` from the same header.
const K_IOHID_ACCESS_TYPE_UNKNOWN: u32 = 2;

/// The tri-state result of `IOHIDCheckAccess` — `Granted`/`Denied` mirror an
/// explicit user answer, `Unknown` means never asked (still promptable via
/// `request_input_monitoring_access`, crate-private).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputMonitoringAccess {
    /// The user granted Input Monitoring access.
    Granted,
    /// The user denied Input Monitoring access.
    Denied,
    /// The user has never been asked.
    Unknown,
}

/// Buffer size for [`tis_current_language`]'s C-string conversion — sized
/// for a BCP-47 language subtag plus terminator (e.g. `"zh-Hans-HK\0"` is 11
/// bytes).
const TIS_LANG_BUF_LEN: isize = 16;

// ── Safe wrappers ─────────────────────────────────────────────────────────

/// Query TIS for the current keyboard input source's primary language tag.
///
/// Returns `Some(tag)` where `tag` is the first language from
/// `kTISPropertyInputSourceLanguages`, or `None` if the source is unavailable.
pub(crate) fn tis_current_language() -> Option<String> {
    use objc2_core_foundation::{CFArray, CFRetained, CFString, CFType};

    // SAFETY: TISCopyCurrentKeyboardInputSource follows the Copy rule (caller
    // owns a +1 reference) — `from_raw` is the correct constructor, and
    // wrapping it as `CFRetained` immediately (instead of holding the raw
    // pointer and releasing it by hand later) means normal `Drop` releases it,
    // with no separate transmute-then-drop step to get wrong.
    let src_ptr = std::ptr::NonNull::new(unsafe { TISCopyCurrentKeyboardInputSource() })?;
    let src: CFRetained<CFType> = unsafe { CFRetained::from_raw(src_ptr.cast()) };

    let lang_array_ptr = unsafe {
        // SAFETY: kTISPropertyInputSourceLanguages is a CFStringRef constant
        // — casting to *mut c_void for the FFI call.
        #[allow(clippy::as_ptr_cast_mut)]
        let key = kTISPropertyInputSourceLanguages.cast::<c_void>();
        TISGetInputSourceProperty(CFRetained::as_ptr(&src).as_ptr().cast(), key)
    };
    let lang_array_ptr = std::ptr::NonNull::new(lang_array_ptr)?;
    // SAFETY: TISGetInputSourceProperty follows the Get rule (+0, borrowed
    // from `src`) per its own doc comment above — `retain` is the correct
    // constructor: it takes our own +1 via a real `CFRetain`, so the later
    // `Drop` releases a reference we actually hold, instead of over-releasing
    // a reference that was never ours. `src` MUST stay alive until this
    // retain completes (the array is only valid while its owner — `src` —
    // is), so it is not dropped until after every use of `lang_array` below.
    let lang_array: CFRetained<CFArray<CFString>> =
        unsafe { CFRetained::retain(lang_array_ptr.cast()) };

    // Get the opaque CFArray to access count and value_at_index.
    let opaque = lang_array.as_opaque();
    let count = opaque.count();
    if count == 0 {
        return None;
    }

    // SAFETY: opaque[count > 0], and value_at_index returns a pointer to
    // the element at the given index (toll-free bridged CFStringRef).
    let ptr = unsafe { opaque.value_at_index(0) };
    let ptr = std::ptr::NonNull::new(ptr.cast_mut())?;

    // SAFETY: ptr points to a CFStringRef *borrowed* from the array (the Get
    // rule: +0, no reference transferred to us) — `retain` is the correct
    // constructor here, NOT `from_raw`. Using `from_raw` (which assumes a +1
    // reference we don't actually hold) would double-release when both this
    // `CFRetained` and the array's own element drop — an over-release bug.
    let cf_str: CFRetained<CFString> = unsafe { CFRetained::retain(ptr.cast()) };

    // Convert CFString to Rust String via UTF-8 C string.
    let encoding = cf_str.fastest_encoding();
    #[allow(clippy::cast_sign_loss)] // TIS_LANG_BUF_LEN is a small positive literal
    let mut buf = [0u8; TIS_LANG_BUF_LEN as usize];
    if unsafe { cf_str.c_string(buf.as_mut_ptr().cast(), TIS_LANG_BUF_LEN, encoding) } {
        // SAFETY: c_string null-terminated the buffer in the given encoding.
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(buf.as_ptr().cast());
            if let Ok(s) = cstr.to_str() {
                return Some(s.to_string());
            }
        }
    }

    None
}

/// Query whether secure event input is active (e.g. a password field is focused).
pub(crate) fn is_secure_event_input_enabled() -> bool {
    unsafe { IsSecureEventInputEnabled() != 0 }
}

/// Query whether the current process has accessibility permission.
pub(crate) fn is_accessibility_trusted() -> bool {
    unsafe { AXIsProcessTrusted() }
}

/// Trigger the native Accessibility permission prompt and add this process to
/// the Accessibility list. Returns whether the process is currently trusted.
///
/// Wraps `AXIsProcessTrustedWithOptions({ kAXTrustedCheckOptionPrompt: true })`.
/// Note: after the user grants access, the process must be **relaunched** for a
/// new `CGEventTap` to bind — a live grant does not retroactively arm the tap.
pub(crate) fn prompt_accessibility_trust() -> bool {
    use objc2_core_foundation::{CFBoolean, CFDictionary, CFString};

    // SAFETY: kAXTrustedCheckOptionPrompt is a non-null CFStringRef framework
    // constant; its value is a valid, immortal CFString.
    let key: &CFString = unsafe { &*kAXTrustedCheckOptionPrompt.cast::<CFString>() };
    let value: &CFBoolean = CFBoolean::new(true);
    let options = CFDictionary::<CFString, CFBoolean>::from_slices(&[key], &[value]);

    // AXIsProcessTrustedWithOptions takes an opaque CFDictionaryRef.
    let options_ptr: *const c_void = std::ptr::addr_of!(*options).cast();
    unsafe { AXIsProcessTrustedWithOptions(options_ptr) }
}

/// Query the current process's Input Monitoring access as the tri-state
/// `IOHIDCheckAccess` actually reports it, rather than collapsing straight
/// to a bool.
///
/// Pure/non-prompting: `IOHIDCheckAccess` never shows a dialog (only
/// `request_input_monitoring_access` does) — this is what lets the
/// preflight permission gate check every grant without side effects.
pub(crate) fn input_monitoring_access() -> InputMonitoringAccess {
    // The documented K_IOHID_ACCESS_TYPE_UNKNOWN arm and the trailing
    // wildcard (any future/unrecognized value) both mean "never asked" —
    // same result, kept as separate arms purely so the header-derived
    // constant stays a named, documented match target rather than a magic
    // number that only appears in a comment.
    #[allow(clippy::match_same_arms)]
    match unsafe { IOHIDCheckAccess(K_IOHID_REQUEST_TYPE_LISTEN_EVENT) } {
        K_IOHID_ACCESS_TYPE_GRANTED => InputMonitoringAccess::Granted,
        K_IOHID_ACCESS_TYPE_DENIED => InputMonitoringAccess::Denied,
        K_IOHID_ACCESS_TYPE_UNKNOWN => InputMonitoringAccess::Unknown,
        _ => InputMonitoringAccess::Unknown,
    }
}

/// Query whether the current process has Input Monitoring permission.
///
/// A thin bool projection of [`input_monitoring_access`] (CONSTITUTION rule
/// 26 — the tri-state check is the single source of truth; this is kept for
/// existing bool-only callers such as `hotkey.rs`'s warn-only check).
pub(crate) fn is_input_monitoring_trusted() -> bool {
    matches!(input_monitoring_access(), InputMonitoringAccess::Granted)
}

/// Trigger the native Input Monitoring permission prompt.
///
/// Fire-and-forget, like `vuho_audio::request_mic_access_async`: the modal
/// system dialog can block on user input arbitrarily long, so the caller
/// doesn't wait for it here — re-check via [`is_input_monitoring_trusted`]
/// on the next poll.
pub(crate) fn request_input_monitoring_access() {
    unsafe {
        let _ = IOHIDRequestAccess(K_IOHID_REQUEST_TYPE_LISTEN_EVENT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secure_input_returns_bool() {
        let _ = is_secure_event_input_enabled();
    }

    #[test]
    fn accessibility_trusted_returns_bool() {
        let _ = is_accessibility_trusted();
    }

    #[test]
    fn input_monitoring_trusted_returns_bool() {
        let _ = is_input_monitoring_trusted();
    }

    #[test]
    fn input_monitoring_access_matches_bool_wrapper() {
        let access = input_monitoring_access();
        assert_eq!(
            access == InputMonitoringAccess::Granted,
            is_input_monitoring_trusted()
        );
    }

    /// Regression test for the TIS languages-array over-release / use-after-free
    /// (the array was wrapped with `CFRetained::from_raw` — assuming a +1
    /// reference `TISGetInputSourceProperty` never grants under its actual
    /// Get-rule contract — and the owning input source was dropped before the
    /// array was read). Calling `tis_current_language` many times in a row
    /// exercises the retain/release pair on every call; a real over-release
    /// corrupts the CF allocator's heap and reliably crashes (or, under
    /// `MallocScribble`, reads back the sentinel corruption pattern) well
    /// before this loop count on a real keyboard input source.
    #[test]
    fn tis_current_language_repeated_calls_do_not_corrupt_memory() {
        for _ in 0..500 {
            let _ = tis_current_language();
        }
    }
}
