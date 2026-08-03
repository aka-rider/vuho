//! Text injection into the focused application.
//!
//! Copies text to the clipboard, then synthesizes a ⌘V keystroke via
//! `CGEvent`. Handles Secure Input (password field) by leaving text on
//! the clipboard and returning a recoverable error (ADR-012).

use std::time::{Duration, Instant};

use objc2_app_kit::NSPasteboard;

use crate::copy_to_clipboard;
use crate::sys;
use objc2_core_graphics::{
    CGEvent, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
};

/// How often to poll `NSPasteboard`'s `changeCount` while confirming the
/// clipboard write actually landed on the pasteboard server (CONSTITUTION
/// rule 32 — named, not a magic number).
const CHANGE_COUNT_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Upper bound on how long to poll `changeCount` before giving up and
/// proceeding anyway. `NSPasteboard` writes are a synchronous IPC call to
/// the pasteboard server, so in practice `changeCount` has already advanced
/// by the time `copy_to_clipboard` returns and this loop exits on its first
/// check — this budget only bounds worst-case latency if that assumption
/// is ever wrong on some macOS version, it never *waits out* a known-slow
/// operation.
const CHANGE_COUNT_POLL_BUDGET: Duration = Duration::from_millis(50);

/// Residual settle delay after the clipboard write is *confirmed* visible
/// via `changeCount`, specifically for third-party clipboard-manager apps
/// (Paste, Raycast, Maccy) that asynchronously poll the pasteboard on their
/// own schedule.
///
/// Honest limitation: there is no event *this process* can observe for "the
/// destination app has noticed the pasteboard change and is ready to
/// receive ⌘V" — that state lives entirely inside another process. This
/// constant remains what it always was (an empirically-chosen 50ms), but is
/// now applied only *after* `wait_for_clipboard_write` has confirmed our
/// own write actually landed, rather than stacked blindly on top of an
/// unconfirmed one (CONSTITUTION rule 32: the part of the wait that *can*
/// be tied to an observable event now is; the part that fundamentally
/// can't stays a documented, bounded compromise instead of a bare sleep).
const CLIPBOARD_MANAGER_SETTLE: Duration = Duration::from_millis(50);

/// Key code for the `V` key (`kVK_ANSI_V`).
const KEYCODE_V: u16 = 9;

/// Inject `text` into the currently focused application.
///
/// Copies `text` to the system clipboard, then synthesizes a ⌘V keystroke
/// to paste it at the cursor position.
///
/// # Secure Input (ADR-012)
///
/// If Secure Input is active (e.g. password field focused), keyboard events
/// are blocked by macOS. In this case the text is left on the clipboard and
/// `Err(OsError::SecureInputActive)` is returned — a recoverable error
/// indicating the user should paste manually.
///
/// # Errors
///
/// - `OsError::ClipboardWrite` — clipboard operation failed.
/// - `OsError::SecureInputActive` — Secure Input is active; text is on clipboard.
/// - `OsError::InjectionFailed` — `CGEvent` synthesis failed.
pub fn inject_text(text: &str) -> Result<(), crate::OsError> {
    // Step a: Check Secure Input first.
    if sys::is_secure_event_input_enabled() {
        copy_to_clipboard(text)?;
        return Err(crate::OsError::SecureInputActive);
    }

    // Step b: Copy to clipboard, capturing the pre-write changeCount so
    // step c can observe our own write actually land instead of guessing.
    let pb = NSPasteboard::generalPasteboard();
    let before = pb.changeCount();
    copy_to_clipboard(text)?;

    // Step c: Confirm the write landed (observable), then a bounded settle
    // for clipboard-manager apps (unobservable — see CLIPBOARD_MANAGER_SETTLE's doc).
    wait_for_clipboard_write(&pb, before);
    std::thread::sleep(CLIPBOARD_MANAGER_SETTLE);

    // Step d: Synthesize ⌘V.
    paste_via_cmd_v()
}

/// Poll `pasteboard`'s `changeCount` until it differs from `before` (our
/// write has landed on the pasteboard server) or [`CHANGE_COUNT_POLL_BUDGET`]
/// elapses, whichever comes first.
///
/// Replaces a blind fixed sleep with an actual observation of the one part
/// of "did the write land" that this process *can* observe (CONSTITUTION
/// rule 32) — `copy_to_clipboard`'s `setString_forType` returning `true`
/// only means the pasteboard server *accepted* the write, not that
/// `changeCount` has necessarily been read back by us yet on every macOS
/// version; polling closes that gap without assuming a fixed duration.
///
/// Thin wrapper over [`poll_until_changed_or_budget`], the pure loop this
/// delegates to — kept separate so the bounded-loop *behavior* is
/// unit-testable against a fake counter instead of the real, process-wide
/// `NSPasteboard` (shared mutable state other tests in this crate also
/// write to, which would make a real-pasteboard timing assertion flaky
/// under `cargo test`'s parallel test threads).
fn wait_for_clipboard_write(pasteboard: &NSPasteboard, before: isize) {
    poll_until_changed_or_budget(
        before,
        CHANGE_COUNT_POLL_BUDGET,
        CHANGE_COUNT_POLL_INTERVAL,
        || pasteboard.changeCount(),
    );
}

/// Pure polling loop: calls `current` repeatedly until it returns a value
/// different from `before`, or `budget` elapses — whichever comes first.
/// See [`wait_for_clipboard_write`] for why this is split out.
fn poll_until_changed_or_budget(
    before: isize,
    budget: Duration,
    interval: Duration,
    mut current: impl FnMut() -> isize,
) {
    let deadline = Instant::now() + budget;
    while current() == before && Instant::now() < deadline {
        std::thread::sleep(interval);
    }
}

/// Synthesize a ⌘V keystroke via `CGEvent` to paste from clipboard.
///
/// Posts key-down and key-up events at the HID event tap level so they
/// reach the frontmost application.
///
/// # Errors
///
/// Returns `OsError::InjectionFailed` if `CGEvent` creation or posting fails.
fn paste_via_cmd_v() -> Result<(), crate::OsError> {
    let source = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
        .ok_or(crate::OsError::InjectionFailed)?;
    let flags = CGEventFlags::MaskCommand;

    // Key down V with Command.
    let key_down = CGEvent::new_keyboard_event(Some(&source), KEYCODE_V, true)
        .ok_or(crate::OsError::InjectionFailed)?;
    CGEvent::set_flags(Some(&key_down), flags);
    CGEvent::post(CGEventTapLocation::HIDEventTap, Some(&key_down));

    // Key up V with Command.
    let key_up = CGEvent::new_keyboard_event(Some(&source), KEYCODE_V, false)
        .ok_or(crate::OsError::InjectionFailed)?;
    CGEvent::set_flags(Some(&key_up), flags);
    CGEvent::post(CGEventTapLocation::HIDEventTap, Some(&key_up));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipboard_manager_settle_is_reasonable() {
        assert!(CLIPBOARD_MANAGER_SETTLE >= Duration::from_millis(20));
        assert!(CLIPBOARD_MANAGER_SETTLE <= Duration::from_millis(200));
    }

    #[test]
    fn change_count_poll_budget_is_bounded_and_coarser_than_its_interval() {
        assert!(CHANGE_COUNT_POLL_INTERVAL < CHANGE_COUNT_POLL_BUDGET);
        assert!(CHANGE_COUNT_POLL_BUDGET <= Duration::from_millis(200));
    }

    /// Falsification target for the "blind sleep" bug: once the pasteboard's
    /// `changeCount` has already advanced past `before` (a real write landed
    /// before this function was even called, as happens in `inject_text`),
    /// `wait_for_clipboard_write` must return on essentially the first
    /// check, not wait out the full budget regardless.
    ///
    /// May be flaky in a headless/CI environment with no working pasteboard
    /// service — tolerates that by not failing on `copy_to_clipboard`'s own
    /// result, only asserting the *timing* once a write is attempted.
    #[test]
    fn wait_for_clipboard_write_returns_promptly_once_the_count_has_already_changed() {
        let pb = NSPasteboard::generalPasteboard();
        let before = pb.changeCount();
        let _ = copy_to_clipboard("vuho-inject-test-changecount");
        if pb.changeCount() == before {
            // No working pasteboard service in this environment — nothing to
            // observe, and looping would just prove the budget path instead
            // (covered by the sibling test below).
            return;
        }
        let started = Instant::now();
        wait_for_clipboard_write(&pb, before);
        assert!(
            started.elapsed() < CHANGE_COUNT_POLL_BUDGET,
            "must return promptly once changeCount has already advanced, got {:?}",
            started.elapsed()
        );
    }

    /// The other half of the falsification target: when the counter
    /// genuinely never changes, the poll must give up at the given budget
    /// rather than spin forever — bounded, not infinite.
    ///
    /// Deliberately drives the pure [`poll_until_changed_or_budget`] with a
    /// fake counter instead of the real, process-wide `NSPasteboard`: other
    /// tests in this crate (`clipboard::tests::clipboard_roundtrip_smoke`,
    /// `tests::clipboard_copy_smoke`) also write to that same shared system
    /// pasteboard, and `cargo test` runs tests in parallel by default — a
    /// timing assertion against the *real* pasteboard's `changeCount` would
    /// be genuinely flaky if one of those raced a write in during this
    /// test's window (CONSTITUTION rule 32: don't let an unrelated shared
    /// resource make a timing test racy when the logic under test doesn't
    /// need to touch it at all).
    #[test]
    fn poll_until_changed_or_budget_is_bounded_when_the_value_never_changes() {
        let budget = Duration::from_millis(30);
        let interval = Duration::from_millis(1);
        let started = Instant::now();
        poll_until_changed_or_budget(0, budget, interval, || 0);
        let elapsed = started.elapsed();
        assert!(
            elapsed >= budget,
            "must wait out the full budget when the value never changes, got {elapsed:?}"
        );
        // Generous slack for scheduler jitter — still proves it's bounded.
        assert!(
            elapsed < budget * 10,
            "must not overrun the budget by more than scheduler jitter, got {elapsed:?}"
        );
    }

    /// The counterpart using a fake counter that *does* change on the first
    /// call — the loop must exit immediately rather than sleep at all.
    #[test]
    fn poll_until_changed_or_budget_exits_immediately_once_changed() {
        let budget = Duration::from_secs(10); // would time the test out if not exited early
        let started = Instant::now();
        poll_until_changed_or_budget(0, budget, Duration::from_millis(1), || 1);
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "must exit on the first check once the value has already changed"
        );
    }
}
