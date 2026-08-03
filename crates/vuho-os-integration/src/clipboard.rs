//! Clipboard write via `NSPasteboard`.
//!
//! Replaces `arboard` with native `NSPasteboard` via objc2.
//! This is the correct choice for `vuho-os-integration` which has no
//! GPUI dependency and is used by the pipeline independently of the overlay.

use objc2_app_kit::{NSPasteboard, NSPasteboardTypeString};
use objc2_foundation::NSString;

/// Copies `text` to the system clipboard using `NSPasteboard`.
///
/// Clears the pasteboard contents first, then writes the text as `NSPasteboardTypeString`.
///
/// # Errors
///
/// Returns `OsError::ClipboardWrite` if `setString_forType` fails.
pub fn copy_to_clipboard(text: &str) -> Result<(), crate::OsError> {
    let pb = NSPasteboard::generalPasteboard();
    pb.clearContents();
    let ns_text = NSString::from_str(text);
    // SAFETY: `setString_forType` itself is a safe method — what actually
    // needs `unsafe` here is referencing `NSPasteboardTypeString`, an
    // `extern static` (a foreign global, so the Rust compiler cannot verify
    // its initialization/aliasing on our behalf per E0133). It is an
    // AppKit-framework-provided `NSPasteboardType` constant, immutable and
    // valid for the process lifetime — reading it is sound.
    let success = unsafe { pb.setString_forType(&ns_text, NSPasteboardTypeString) };
    if success {
        Ok(())
    } else {
        Err(crate::OsError::ClipboardWrite)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clipboard roundtrip smoke test.
    ///
    /// May fail headless; assert it doesn't panic.
    #[test]
    fn clipboard_roundtrip_smoke() {
        let _ = copy_to_clipboard("vuho-test-1234");
    }
}
