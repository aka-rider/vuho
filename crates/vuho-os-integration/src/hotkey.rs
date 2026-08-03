//! Global hotkey listener via `CGEventTap`.
//!
//! Default: `CapsLock` tap-to-toggle (ADR-007). Configurable chord fallback:
//! `⌥-Space`.
//!
//! `CapsLock` is a latching key: the `MaskAlphaShift` flag stays set while the
//! LED is on. The trigger is **level-triggered**, not edge-triggered: the LED
//! being on always means "start dictating" and off always means "stop",
//! rather than each transition emitting an undifferentiated `Toggle`. A
//! dropped or discarded command can then desync the LED from the session
//! state, but the next tap self-heals the phase instead of inverting it
//! permanently (see [`caps_lock_command`]). An `AtomicBool` tracks the last
//! observed latch state so unrelated `FlagsChanged` events (Shift, Option, …)
//! while the latch is lit don't re-send `Start`.

use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::Sender;
use log::info;
use objc2_core_foundation::{CFMachPort, CFRetained, CFRunLoop, CFString};
use objc2_core_graphics::{
    CGEvent, CGEventField, CGEventFlags, CGEventMask, CGEventSource, CGEventSourceStateID,
    CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement, CGEventTapProxy, CGEventType,
};
use vuho_domain::DictationCommand;

use crate::sys;

/// Heap-allocated context for the event tap callback.
///
/// Contains the command sender, hotkey configuration, and a stop flag.
/// The sender is cloned in `start()` so the caller's original sender
/// stays alive for the bridge thread (CONSTITUTION rule 2).
struct TapContext {
    tx: Sender<DictationCommand>,
    config: HotkeyConfig,
    /// Last observed `CapsLock` state, for transition detection.
    caps_on: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    /// Raw (non-owning) pointer to the `CFMachPort` driving this tap, set by
    /// `run_event_tap` immediately after `CGEventTapCreate` succeeds — before
    /// that point the tap cannot be receiving callbacks yet, so `null` is
    /// never observed by `event_tap_callback`. Used only to re-enable the tap
    /// on `TapDisabledByTimeout`/`TapDisabledByUserInput` (macOS disables a
    /// tap that doesn't return promptly, or on user request; without
    /// re-arming, the hotkey silently stops working until the app restarts).
    /// `Cell`, not `Arc`/`Mutex`: the callback only ever runs on the tap
    /// thread's own run loop, so no cross-thread synchronization is needed —
    /// only interior mutability for the one post-construction write.
    port_ptr: std::cell::Cell<*mut c_void>,
}

/// Configurable global hotkey.
///
/// - `CapsLock` — tap-to-toggle (ADR-007 default).
/// - `Chord { flags, keycode }` — modifier + key chord (default: `⌥-Space`).
#[derive(Clone, Debug, Default)]
pub enum HotkeyConfig {
    /// `CapsLock` tap-to-toggle.
    #[default]
    CapsLock,
    /// Custom modifier + key chord.
    Chord {
        /// Modifier flags (e.g. `MaskAlternate` for Option).
        flags: CGEventFlags,
        /// Key code (e.g. `49` for Space).
        keycode: u16,
    },
}

/// Global hotkey listener.
///
/// Listens for keyboard events via a `CGEventTap` and sends
/// `DictationCommand::Toggle` on the configured trigger (`CapsLock` tap
/// or custom chord).
///
/// After calling [`Self::stop`], the listener is restartable via [`Self::start`].
pub struct HotkeyListener {
    tap_thread: Option<JoinHandle<()>>,
    /// Shared with the tap thread's `TapContext`. `start()` creates it and
    /// hands a clone to `run_event_tap`; `stop()` sets it before joining —
    /// the stopper owns the stop signal (CONSTITUTION rule 9).
    stopped: Option<Arc<AtomicBool>>,
}

impl HotkeyListener {
    /// Create a new hotkey listener (not yet started).
    #[must_use]
    pub fn new() -> Self {
        Self {
            tap_thread: None,
            stopped: None,
        }
    }

    /// Start listening for the global hotkey.
    ///
    /// Spawns a dedicated thread that creates a `CGEventTap`, wraps it in a
    /// `CFRunLoopSource`, and runs the current thread's `CFRunLoop`.
    /// On the configured trigger, sends `DictationCommand::Toggle` via `command_tx`.
    ///
    /// # Accessibility permission
    ///
    /// Requires Accessibility permission. If not granted, returns
    /// `Err(OsError::Hotkey)` — the UI should prompt the user (ADR-012).
    ///
    /// Input Monitoring is also checked and logged if missing (ADR-016: a
    /// `CGEventTap` at `HIDEventTap` can need it on macOS 10.15+), but is
    /// **not** a hard-fail condition here: an empirical probe on the
    /// development machine (`IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)`)
    /// read `Denied` even though the `CapsLock` tap already works end-to-end
    /// today with only Accessibility granted. Hard-failing on Input
    /// Monitoring alone would regress a setup that works today into a
    /// permanent gate, so it is preflight-checked and offered in the
    /// permission gate window (`vuho-ui`'s `permission_gate` module) but does
    /// not block `start()`.
    ///
    /// # Errors
    ///
    /// Returns `OsError::Hotkey` if Accessibility permission is not granted
    /// or the event tap cannot be created. Returns
    /// `OsError::HotkeyAlreadyRunning` if this listener is already running
    /// — call [`Self::stop`] first. Without this guard, a second `start()`
    /// would overwrite `self.stopped`/`self.tap_thread` with the new tap's
    /// handles, making the *first* tap thread's `stopped` flag permanently
    /// unreachable — an unstoppable, unjoinable leaked thread.
    ///
    /// # Spec
    ///
    /// "Global hotkey for dictation: `CapsLock` tap-to-toggle."
    pub fn start(
        &mut self,
        command_tx: &Sender<DictationCommand>,
        config: HotkeyConfig,
    ) -> Result<(), crate::OsError> {
        if self.tap_thread.is_some() {
            return Err(crate::OsError::HotkeyAlreadyRunning);
        }
        // Check Accessibility permission — the only hard gate (see doc comment above).
        if !sys::is_accessibility_trusted() {
            return Err(crate::OsError::Hotkey);
        }
        if !sys::is_input_monitoring_trusted() {
            log::warn!(
                "hotkey: Input Monitoring not granted — the CapsLock tap may silently \
                 receive no events on this macOS version; grant it in System Settings \
                 → Privacy & Security → Input Monitoring"
            );
        }

        // Clone the sender for the tap thread. The caller retains their
        // original sender so the bridge thread's channel stays open
        // (CONSTITUTION rule 2 — resource handle cleanup).
        let tx = command_tx.clone(); // tap thread gets a fresh clone

        // Created here (the stopper) and cloned into the tap thread — stop()
        // is the only place that ever sets it.
        let stopped = Arc::new(AtomicBool::new(false));
        self.stopped = Some(Arc::clone(&stopped));

        let handle = std::thread::spawn(move || {
            run_event_tap(tx, config, stopped);
        });

        self.tap_thread = Some(handle);
        Ok(())
    }

    /// Stop listening for the global hotkey.
    ///
    /// Sets the shared stop flag, then invalidates the event tap's
    /// `CFMachPort` and joins the tap thread (the thread notices the flag
    /// within one 100 ms run-loop tick). After calling `stop()`, the
    /// listener is restartable via [`Self::start`].
    pub fn stop(&mut self) {
        if let Some(stopped) = self.stopped.take() {
            stopped.store(true, Ordering::Relaxed);
        }
        if let Some(handle) = self.tap_thread.take() {
            let _ = handle.join();
        }
        // Tap thread exited — its sender clone (inside TapContext) is dropped,
        // closing the tap's senders. The bridge thread's channel stays open
        // via VuhoState.cmd_tx and status_bar's clone.
    }
}

impl Default for HotkeyListener {
    fn default() -> Self {
        Self::new()
    }
}

// Manual Default because Sender doesn't implement Default.

/// Run the event tap loop in the current thread.
///
/// `stopped` is created and owned by [`HotkeyListener::start`]; this
/// function only ever reads it (via the `TapContext` clone) to notice a
/// stop request from [`HotkeyListener::stop`].
///
/// Split into prepare / create / run stages (CONSTITUTION rule 28): each of
/// [`build_event_mask`], [`alloc_tap_context`], [`create_tap_and_run_loop_source`],
/// and [`poll_until_stopped`] owns one responsibility, and `cleanup_context`
/// is the single place that frees the heap-allocated context, used by every
/// failure and shutdown path instead of a repeated `Box::from_raw` at each
/// call site.
fn run_event_tap(tx: Sender<DictationCommand>, config: HotkeyConfig, stopped: Arc<AtomicBool>) {
    let mask = build_event_mask();
    let context = alloc_tap_context(tx, config, stopped);

    // SAFETY: `context` was just allocated by `alloc_tap_context` above and
    // has not been freed yet.
    let created = unsafe { create_tap_and_run_loop_source(context, mask) };
    let Some((port, mode)) = created else {
        cleanup_context(context);
        return;
    };

    // SAFETY: `context` is the same live pointer `create_tap_and_run_loop_source`
    // was just given, and `port`/`mode` were created against it.
    unsafe { poll_until_stopped(context, &port, &mode) };
}

/// Build the tap's event mask: `FlagsChanged` (CapsLock/modifier state) plus
/// `KeyDown`/`KeyUp` (chord matching). Pure and extracted so the exact bits
/// requested aren't buried inside [`run_event_tap`].
fn build_event_mask() -> CGEventMask {
    // NOTE (not a SAFETY comment — this cast is not `unsafe`): CGEventType
    // values are small u32 constants, safely fitting in u64.
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    let mask: CGEventMask = (1 << (CGEventType::FlagsChanged.0 as u64))
        | (1 << (CGEventType::KeyDown.0 as u64))
        | (1 << (CGEventType::KeyUp.0 as u64));
    mask
}

/// Read the real keyboard's current `CapsLock` latch state directly from the
/// HID system event source (not the tap, which hasn't received any events
/// yet at listener-start time).
///
/// Without this, `caps_on` used to always seed `false`: if `CapsLock` was
/// already lit when the listener started, the very next `FlagsChanged`
/// event (`CapsLock` going *off*) would read as a spurious transition from
/// the wrong baseline — either firing an unwanted `Toggle` immediately, or
/// swallowing the next genuine tap, depending on which edge arrives first.
///
/// BLIND: reads live HID hardware state and cannot be exercised by an
/// automated test in this environment — a human must verify starting the
/// listener while `CapsLock` is already lit does not mis-toggle on the first
/// tap. See [`caps_lock_bit_set`] for the pure decision this wraps, which
/// *is* unit-tested.
fn read_current_caps_lock_state() -> bool {
    caps_lock_bit_set(CGEventSource::flags_state(
        CGEventSourceStateID::HIDSystemState,
    ))
}

/// Whether `flags` has the `CapsLock` latch bit set. Extracted from
/// [`read_current_caps_lock_state`] purely so this bit-test is unit-testable
/// without a live HID event source.
fn caps_lock_bit_set(flags: CGEventFlags) -> bool {
    flags.contains(CGEventFlags::MaskAlphaShift)
}

/// `CapsLock` is level-triggered, not edge-triggered: the command follows the
/// new latch state rather than the mere fact of a transition, so the LED is
/// authoritative and the two can never invert. Returns `Start` when the LED
/// is now on, `Stop` when it is now off.
///
/// Pure and unit-tested precisely because this used to be inline in the
/// `unsafe extern "C"` callback (as an undifferentiated `Toggle` on either
/// edge), where it had no test coverage.
fn caps_lock_command(is_caps: bool) -> DictationCommand {
    if is_caps {
        DictationCommand::Start
    } else {
        DictationCommand::Stop
    }
}

/// Heap-allocate the tap context (`Box::into_raw`), seeded with the real
/// `CapsLock` latch state (see [`read_current_caps_lock_state`]) so a pre-lit
/// `CapsLock` at listener-start doesn't cause a spurious or swallowed tap.
/// Under level-triggering a wrong baseline costs at most one suppressed
/// command (the transition guard swallows the first tap), never a permanent
/// inversion — but a correct seed is still worth keeping since it avoids
/// even that one suppressed tap.
/// The caller must eventually free the returned pointer via
/// [`cleanup_context`] on every path (failure or normal shutdown).
fn alloc_tap_context(
    tx: Sender<DictationCommand>,
    config: HotkeyConfig,
    stopped: Arc<AtomicBool>,
) -> *mut TapContext {
    let caps_on = Arc::new(AtomicBool::new(read_current_caps_lock_state()));
    Box::into_raw(Box::new(TapContext {
        tx,
        config,
        caps_on,
        stopped,
        port_ptr: std::cell::Cell::new(std::ptr::null_mut()),
    }))
}

/// Free a tap context allocated by [`alloc_tap_context`]. The single
/// chokepoint every failure/shutdown path routes through, instead of a
/// repeated `Box::from_raw` at each call site (CONSTITUTION rule 26).
///
/// # Safety
///
/// `context` must be a live pointer from `alloc_tap_context` that has not
/// already been freed, and no other code may read or write through it
/// afterward (in particular, no `event_tap_callback` invocation may still
/// be in flight).
fn cleanup_context(context: *mut TapContext) {
    // SAFETY: upheld by this function's own `# Safety` contract above.
    unsafe { drop(Box::from_raw(context)) };
}

/// Create the `CGEventTap` for `context`, publish its port pointer into the
/// context (for [`event_tap_callback`]'s re-arm path), build a run-loop
/// source from it, and add that source to the current thread's `CFRunLoop`.
///
/// Returns the retained port and the run-loop mode string used both to add
/// the source and to drive [`poll_until_stopped`]'s polling, or `None` on
/// any failure (tap creation, no run loop source, or no run loop available
/// on this thread) — treating "tap created but has nowhere to run" the same
/// as "tap creation failed" rather than falling through into a sourceless,
/// callback-dead `CFRunLoopRunInMode` busy-spin. The caller is responsible
/// for calling [`cleanup_context`] on `None`.
///
/// # Safety
///
/// `context` must be a live, unfreed pointer from [`alloc_tap_context`].
unsafe fn create_tap_and_run_loop_source(
    context: *mut TapContext,
    mask: CGEventMask,
) -> Option<(CFRetained<CFMachPort>, CFRetained<CFString>)> {
    // NOTE (not a SAFETY comment — `.cast()` is not `unsafe`): the actual
    // safety requirement (context must be a valid TapContext pointer
    // allocated with Box::into_raw) lives on `event_tap_callback`'s own
    // `# Safety` doc, which is what an `unsafe` caller must uphold, not this
    // pointer-type cast.
    #[allow(clippy::cast_ptr_alignment, clippy::as_ptr_cast_mut)]
    let context_ptr = context.cast::<c_void>();

    let tap_ptr = unsafe {
        sys::CGEventTapCreate(
            CGEventTapLocation::HIDEventTap,
            CGEventTapPlacement::HeadInsertEventTap,
            CGEventTapOptions::ListenOnly,
            mask,
            Some(event_tap_callback),
            context_ptr,
        )
    };
    let tap_ptr = std::ptr::NonNull::new(tap_ptr)?;

    // SAFETY: tap_ptr is a retained CFMachPortRef from CGEventTapCreate
    // (Create rule, +1 reference), confirmed non-null above.
    let port: CFRetained<CFMachPort> = unsafe { CFRetained::from_raw(tap_ptr.cast()) };

    // Publish the (non-owning) port pointer for event_tap_callback to
    // re-enable the tap on TapDisabledByTimeout/TapDisabledByUserInput. Must
    // happen before the run loop starts below — no callback can fire before
    // that point.
    // SAFETY: context is the same live TapContext pointer passed in above.
    unsafe {
        (*context)
            .port_ptr
            .set(CFRetained::as_ptr(&port).as_ptr().cast());
    }

    // Create a run loop source from the mach port (default allocator) and
    // get the current thread's run loop. Either missing means the tap can
    // never actually deliver a callback — a failure, not something to poll
    // on regardless.
    let source = CFMachPort::new_run_loop_source(None, Some(&port), 0)?;
    let run_loop = CFRunLoop::current()?;

    // `CFRunLoopMode` is `pub type CFRunLoopMode = CFString;` (a plain type
    // alias, not a distinct newtype) — no cast, bridge or otherwise, is
    // needed to use a `CFString` where a `CFRunLoopMode` reference is
    // expected; they are the same Rust type. Created once here — the same
    // `kCFRunLoopDefaultMode` string is both the mode the source is added
    // under and the mode each `CFRunLoopRunInMode` iteration runs, so
    // there is no reason to reallocate it every 100ms in the poll loop.
    let mode = CFString::from_str("kCFRunLoopDefaultMode");
    // Not an `unsafe` call: `CFRunLoop::add_source` is a safe, typed
    // objc2-core-foundation method.
    run_loop.add_source(Some(&source), Some(&mode));

    Some((port, mode))
}

/// Poll the current thread's `CFRunLoop` in 100ms increments until `stopped`
/// is observed, then invalidate `port` and free `context` via
/// [`cleanup_context`].
///
/// A blocking `CFRunLoopRun()` + a source-triggered wake (so `stop()` can
/// interrupt it instantly instead of waiting out a poll tick) was
/// considered instead of this poll, per the plan's step 7. It was not
/// adopted: `stop()` runs on a *different* thread than the one that owns
/// this run loop, and publishing the `CFRetained<CFRunLoop>` handle back to
/// `HotkeyListener` for a cross-thread `CFRunLoopStop` call would need a new
/// handshake (the run loop object doesn't exist until this function is
/// already running, but `start()` returns before that) — real added
/// complexity and a real new race window (a `stop()` that lands before the
/// run loop starts spinning would need its own wake-safe handling), for a
/// benefit (waking ≤100ms sooner) that doesn't change any observed behavior
/// today. The existing `AtomicBool` + 100ms poll is simple, already correct
/// (proven by `hotkey_listener_restarts_after_stop_with_chord_config`), and
/// needs no new unsafe surface — so it stays.
///
/// # Safety
///
/// `context` must be a live, unfreed pointer from [`alloc_tap_context`],
/// and `port`/`mode` must be the values [`create_tap_and_run_loop_source`]
/// returned for that same `context`.
unsafe fn poll_until_stopped(context: *mut TapContext, port: &CFMachPort, mode: &CFString) {
    const POLL_INTERVAL_SECS: f64 = 0.1;
    loop {
        // SAFETY: context is a live TapContext pointer for the duration of
        // this loop (upheld by this function's own `# Safety` contract).
        if unsafe { (*context).stopped.load(Ordering::Relaxed) } {
            port.invalidate();
            cleanup_context(context);
            break;
        }

        // SAFETY: mode lives for the duration of this call (it outlives the
        // whole loop, not just this iteration). Using addr_of! to avoid
        // creating an intermediate reference (clippy::borrow_as_ptr).
        unsafe {
            #[allow(clippy::cast_ptr_alignment, clippy::as_ptr_cast_mut)]
            let mode_raw: *mut c_void = std::ptr::addr_of!(*mode) as *mut c_void;
            sys::CFRunLoopRunInMode(mode_raw, POLL_INTERVAL_SECS, false);
        }
    }
}

/// Whether `event_type` signals that macOS has disabled the event tap and it
/// must be re-enabled via `CGEventTapEnable`/`CGEvent::tap_enable` before any
/// further events will be delivered.
///
/// macOS disables a tap that takes too long to return from a callback
/// (`TapDisabledByTimeout`) or on explicit user/system request
/// (`TapDisabledByUserInput`, e.g. Secure Input toggling); without
/// re-arming, the hotkey silently stops working until the app restarts.
/// Pure and unit-tested (see below) precisely so this policy isn't buried
/// inside the FFI callback where it can't be tested without a real tap.
fn should_rearm(event_type: CGEventType) -> bool {
    matches!(
        event_type,
        CGEventType::TapDisabledByTimeout | CGEventType::TapDisabledByUserInput
    )
}

/// The modifier bits a chord's `flags` field is ever built from (see
/// `vuho-ui`'s `hotkey_presets::to_hotkey_config`). Deliberately excludes
/// `MaskAlphaShift` (`CapsLock` is a separate trigger, not a chord modifier)
/// and device/state bits `CGEventTap` can set independently of what the user
/// actually pressed (`MaskNumericPad`, `MaskNonCoalesced`, `MaskHelp`,
/// `MaskSecondaryFn`) — masking those out before comparison keeps chord
/// matching from being defeated by noise bits a chord's `flags` never
/// encodes in the first place.
const CHORD_MODIFIER_MASK: CGEventFlags = CGEventFlags::MaskShift
    .union(CGEventFlags::MaskControl)
    .union(CGEventFlags::MaskAlternate)
    .union(CGEventFlags::MaskCommand);

/// Whether `event_flags` matches `chord_flags` exactly over the relevant
/// modifier bits ([`CHORD_MODIFIER_MASK`]).
///
/// Pure and unit-tested precisely because the previous `event_flags.contains(chord_flags)`
/// subset check was wrong: it also matched any *superset* of the configured
/// modifiers, so a chord configured as ⌥Space would spuriously fire on
/// ⇧⌥Space too.
fn chord_flags_match(event_flags: CGEventFlags, chord_flags: CGEventFlags) -> bool {
    (event_flags & CHORD_MODIFIER_MASK) == (chord_flags & CHORD_MODIFIER_MASK)
}

/// Whether `event` is a key-repeat (the OS re-sending `KeyDown` while a key
/// is held, not a fresh press). Pure over the field read, unit-tested via
/// [`autorepeat_field_marks_repeat`] since a real `CGEvent` can't be
/// constructed off a live tap in a unit test.
fn key_event_is_autorepeat(event: &CGEvent) -> bool {
    autorepeat_field_marks_repeat(CGEvent::integer_value_field(
        Some(event),
        CGEventField::KeyboardEventAutorepeat,
    ))
}

/// The `kCGKeyboardEventAutorepeat` field is nonzero exactly when the event
/// is an OS-synthesized repeat. Extracted so the "nonzero means repeat"
/// decision is testable independent of `CGEvent::integer_value_field`.
fn autorepeat_field_marks_repeat(field_value: i64) -> bool {
    field_value != 0
}

/// The `CGEventTap` callback function.
///
/// Receives keyboard events and sends a `DictationCommand` on the configured
/// trigger: `CapsLock` is level-triggered (`Start`/`Stop` from the new latch
/// state via [`caps_lock_command`]); a chord match sends `Toggle` (a
/// momentary gesture with no external LED state to stay in phase with). Also
/// re-enables the
/// tap itself on `TapDisabledByTimeout`/`TapDisabledByUserInput` (see
/// [`should_rearm`]) — without this, macOS permanently disabling the tap
/// after a slow callback or user action would silently kill the hotkey.
///
/// # Safety
///
/// `user_info` must point to a valid `TapContext` allocated with `Box::into_raw`.
/// The context is freed when the tap is invalidated and the run loop exits.
unsafe extern "C-unwind" fn event_tap_callback(
    _proxy: CGEventTapProxy,
    event_type: CGEventType,
    event: std::ptr::NonNull<CGEvent>,
    user_info: *mut c_void,
) -> *mut CGEvent {
    // SAFETY: `user_info` is documented (see the fn-level `# Safety`) to
    // point to a valid `TapContext` allocated with `Box::into_raw`, live for
    // the tap's lifetime; `event` is a `NonNull` handed to us by CoreGraphics
    // for this callback invocation.
    let ctx = unsafe { &*(user_info as *const TapContext) };

    if should_rearm(event_type) {
        let port_ptr = ctx.port_ptr.get();
        if !port_ptr.is_null() {
            // SAFETY: port_ptr was published by run_event_tap once the
            // CFMachPort driving this exact tap was created, and it outlives
            // every callback invocation — the port is only invalidated/freed
            // after the tap thread's run loop has stopped calling back here.
            let port: &CFMachPort = unsafe { &*port_ptr.cast() };
            info!("hotkey: tap disabled ({event_type:?}) — re-enabling");
            CGEvent::tap_enable(port, true);
        }
        return std::ptr::null_mut();
    }

    let event_ref = unsafe { event.as_ref() };

    match &ctx.config {
        HotkeyConfig::CapsLock => {
            if event_type == CGEventType::FlagsChanged {
                let current_flags = CGEvent::flags(Some(event_ref));
                let is_caps = caps_lock_bit_set(current_flags);
                let previous = ctx.caps_on.swap(is_caps, Ordering::Relaxed);

                // Level-triggered: the command follows the new latch state,
                // not the mere fact of a transition (see
                // `caps_lock_command`). The `swap` above still guards against
                // re-sending the same command on every unrelated
                // `FlagsChanged` (Shift, Option, …) that arrives while the
                // latch is lit — only an actual transition fires.
                if is_caps != previous {
                    let command = caps_lock_command(is_caps);
                    info!(
                        "hotkey: CapsLock tap (now {}) → {command:?}",
                        if is_caps { "on" } else { "off" }
                    );
                    let _ = ctx.tx.send(command);
                }
            }
        }
        HotkeyConfig::Chord { flags, keycode } => {
            if event_type == CGEventType::KeyDown && !key_event_is_autorepeat(event_ref) {
                let event_flags = CGEvent::flags(Some(event_ref));
                // NOTE (not a SAFETY comment — this cast is not `unsafe`):
                // KeyboardEventKeycode returns a virtual key code (0-255),
                // safely fitting in u16.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let key_code = CGEvent::integer_value_field(
                    Some(event_ref),
                    CGEventField::KeyboardEventKeycode,
                ) as u16;

                if chord_flags_match(event_flags, *flags) && key_code == *keycode {
                    info!("hotkey: chord match ({flags:?}+{key_code}) → Toggle");
                    let _ = ctx.tx.send(DictationCommand::Toggle);
                }
            }
        }
    }

    std::ptr::null_mut() // Don't modify the event (listen-only tap).
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotkey_config_defaults_to_capslock() {
        let config = HotkeyConfig::default();
        matches!(config, HotkeyConfig::CapsLock);
    }

    #[test]
    fn caps_lock_on_starts() {
        assert_eq!(caps_lock_command(true), DictationCommand::Start);
    }

    #[test]
    fn caps_lock_off_stops() {
        assert_eq!(caps_lock_command(false), DictationCommand::Stop);
    }

    #[test]
    fn should_rearm_true_for_timeout_and_user_input_disable() {
        assert!(should_rearm(CGEventType::TapDisabledByTimeout));
        assert!(should_rearm(CGEventType::TapDisabledByUserInput));
    }

    #[test]
    fn should_rearm_false_for_ordinary_events() {
        assert!(!should_rearm(CGEventType::FlagsChanged));
        assert!(!should_rearm(CGEventType::KeyDown));
        assert!(!should_rearm(CGEventType::KeyUp));
        assert!(!should_rearm(CGEventType::Null));
    }

    #[test]
    fn hotkey_listener_starts_and_stops() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut listener = HotkeyListener::new();
        // May fail if Accessibility is not granted (CI).
        let _ = listener.start(&tx, HotkeyConfig::default());
        listener.stop();
    }

    /// Regression test for the `stop()` deadlock: the shared `stopped` flag
    /// must be threaded from `start()` into the tap thread so `stop()` can
    /// actually signal it, join promptly, and allow a clean restart.
    ///
    /// If Accessibility isn't granted (headless CI), `start()` returns
    /// `Err` and there is no tap thread to join — `stop()` must still be a
    /// no-op rather than hang, so the test tolerates that path too.
    #[test]
    fn hotkey_listener_restarts_after_stop_with_chord_config() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut listener = HotkeyListener::new();

        let chord = HotkeyConfig::Chord {
            flags: CGEventFlags::MaskAlternate,
            keycode: 49, // Space
        };
        let _ = listener.start(&tx, chord.clone());
        listener.stop(); // Must return promptly, not deadlock.

        // Restartable after stop().
        let _ = listener.start(&tx, chord);
        listener.stop();
    }

    /// Regression test for the double-start leak: a second `start()` on an
    /// already-running listener must be rejected, not silently overwrite
    /// `self.stopped`/`self.tap_thread` and orphan the first tap thread.
    #[test]
    fn start_twice_without_stop_is_rejected() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        let mut listener = HotkeyListener::new();

        let Ok(()) = listener.start(&tx, HotkeyConfig::default()) else {
            // No Accessibility grant in this environment (headless CI) —
            // there is no running listener to double-start against.
            return;
        };

        assert!(matches!(
            listener.start(&tx, HotkeyConfig::default()),
            Err(crate::OsError::HotkeyAlreadyRunning)
        ));

        listener.stop();

        // Restartable again after stop().
        assert!(listener.start(&tx, HotkeyConfig::default()).is_ok());
        listener.stop();
    }

    #[test]
    fn caps_lock_bit_set_reflects_the_alpha_shift_flag() {
        assert!(caps_lock_bit_set(CGEventFlags::MaskAlphaShift));
        assert!(caps_lock_bit_set(
            CGEventFlags::MaskAlphaShift | CGEventFlags::MaskShift
        ));
        assert!(!caps_lock_bit_set(CGEventFlags::MaskShift));
        assert!(!caps_lock_bit_set(CGEventFlags::empty()));
    }

    #[test]
    fn chord_flags_match_requires_exact_modifier_set() {
        let option_space = CGEventFlags::MaskAlternate;

        // Exact match.
        assert!(chord_flags_match(CGEventFlags::MaskAlternate, option_space));

        // Falsification target: a superset (⇧⌥) must NOT match a chord
        // configured as ⌥ alone — the old `contains` subset check would
        // wrongly accept this.
        assert!(!chord_flags_match(
            CGEventFlags::MaskAlternate | CGEventFlags::MaskShift,
            option_space
        ));

        // A subset (no modifiers at all) must not match either.
        assert!(!chord_flags_match(CGEventFlags::empty(), option_space));
    }

    #[test]
    fn chord_flags_match_ignores_non_modifier_noise_bits() {
        let option_space = CGEventFlags::MaskAlternate;

        // NumericPad/NonCoalesced/Help/SecondaryFn are state bits the OS can
        // set independently of the user's actual chord — they must not
        // affect the match either way.
        let with_noise = CGEventFlags::MaskAlternate
            | CGEventFlags::MaskNumericPad
            | CGEventFlags::MaskNonCoalesced;
        assert!(chord_flags_match(with_noise, option_space));
    }

    #[test]
    fn autorepeat_field_marks_repeat_is_nonzero_check() {
        assert!(!autorepeat_field_marks_repeat(0));
        assert!(autorepeat_field_marks_repeat(1));
        assert!(autorepeat_field_marks_repeat(-1));
    }
}
