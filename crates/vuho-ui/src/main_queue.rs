//! The one chokepoint for deferring an `AppKit` call off the current call
//! stack, onto the next turn of the main dispatch queue (CONSTITUTION rule
//! 27).
//!
//! **The rule this enforces:** an `AppKit` call that either delivers
//! synchronous delegate callbacks into gpui (e.g. `windowDidMove:` /
//! `setFrameSize:` fired synchronously by `setFrame:display:YES`,
//! `orderFront:`, `orderOut:`, `makeKeyAndOrderFront:`) or pumps a nested run
//! loop (`NSAlert::runModal()`, `NSStatusItem`'s `performClick:`
//! menu-tracking loop) must be issued through [`defer`], never inline. Both
//! classes re-enter gpui while an `App` borrow may still be on the stack —
//! the first via `AsyncApp::update_window`'s non-panicking
//! `try_borrow_mut` (silently dropped, `gpui-0.2.2/src/app/async_context.rs:83-88`),
//! the second via `AsyncApp::update`/`update_entity`'s panicking
//! `borrow_mut()` (`async_context.rs:56, 142-146`). [`defer`] lets the
//! current call stack — and with it any live `App` borrow — unwind first, so
//! the `AppKit` call and whatever it re-enters run against a guaranteed-clean
//! borrow stack.
//!
//! Implemented via `dispatch2::DispatchQueue::main().exec_async`, which
//! guarantees FIFO ordering across every deferred closure — the same
//! ordering callers relied on when each call was issued inline.

use dispatch2::DispatchQueue;

/// Run `f` on the next main-queue turn, after the current call stack — and
/// with it any live gpui `App` borrow — has unwound.
///
/// `f` must be `Send` per `dispatch2`'s own bound, but `AppKit` handles
/// (`Retained<AnyObject>`/`Retained<NSStatusItem>` and friends) are not —
/// they carry no `Sync` impl, so `Retained<T>: Send` never holds for them
/// (`objc2::rc::Retained`'s blanket impl requires `T: Sync + Send`). Callers
/// satisfy `defer`'s bound the same way every call site here does: capture
/// only `Send` primitives (`f64`, `bool`, …) in `f`, and re-derive any
/// `AppKit` handle *inside* the deferred body from a main-thread
/// `thread_local` (`window_config`'s retained `NSWindow`, `status_bar`'s
/// `STATE`) — since `f` only ever runs once dispatched back onto the main
/// thread, reading a main-thread-only `thread_local` from inside it is
/// exactly as safe as reading it from the call site that scheduled it.
pub(crate) fn defer(f: impl FnOnce() + Send + 'static) {
    DispatchQueue::main().exec_async(f);
}
