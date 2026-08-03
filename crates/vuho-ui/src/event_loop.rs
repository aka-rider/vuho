//! Event drains: pipeline (or demo) events → the panel's overlay entity, and
//! status-bar `UiCommand`s → panel transitions. Split out of `main.rs`
//! (WP10) — this module owns the poll-and-apply loops and the pure
//! hide/stale-detection logic they depend on. Retargeted from
//! `WindowHandle<OverlayModel>` to `WindowHandle<PanelRoot>` in WP6
//! (ARCHITECTURE.md ADR-021): events forward into `panel.overlay`, and what
//! used to be direct `window_config::order_front`/`order_out` calls are now
//! `panel::on_session_started`/`panel::hide_if_hud` — the presentation-aware
//! transitions that keep the Hud/Full split intact.

use std::time::Instant;

use gpui::{App, WindowHandle};
use vuho_domain::{DictationEvent, ErrorKind};

use crate::panel::{self, PanelRoot};
use crate::{overlay, permissions};

/// A handle to the production `StatusModel` entity — `()` under
/// `--features demo`, which has no `StatusModel` at all (no menu bar, no
/// settings). One cfg-gated type, rather than cfg-gating every
/// `spawn_event_drain`/`apply_events` call site individually: those
/// functions are shared between production and demo, so they take this by
/// value/reference unconditionally and [`set_recording`] is the sole place
/// that branches on which build it actually is.
#[cfg(not(feature = "demo"))]
pub(crate) type StatusHandle = gpui::Entity<crate::app_status::StatusModel>;
#[cfg(feature = "demo")]
pub(crate) type StatusHandle = ();

/// Reflect a session's recording state in the `StatusModel` (production) —
/// a no-op under `--features demo`, which has no model to update. The sole
/// `#[cfg]` branch point for `apply_events`' three call sites, instead of
/// gating each individually.
#[cfg(not(feature = "demo"))]
fn set_recording(status: &StatusHandle, recording: bool, cx: &mut gpui::AsyncApp) {
    let _ = status.update(cx, |model, cx| {
        model.recording = recording;
        cx.notify();
    });
}

// `StatusHandle` is `()` under this feature — a real 0-byte type, so
// pass-by-reference genuinely is pointlessly indirect here; taking it by
// value instead would require the production arm above (a pointer-sized
// `Entity<StatusModel>`) to diverge on parameter-passing convention between
// the two `#[cfg]` arms of one shared signature, which is worse than one
// pedantic lint suppression.
#[cfg(feature = "demo")]
#[allow(clippy::trivially_copy_pass_by_ref)]
fn set_recording(_status: &StatusHandle, _recording: bool, _cx: &mut gpui::AsyncApp) {}

/// Poll interval for both the demo and production event drains (~60 Hz).
pub(crate) const DRAIN_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);

/// Drain all currently pending events from `rx`.
///
/// Returns `Some(events)` (possibly empty) while the channel is open, or
/// `None` once the sender has been dropped — the caller should stop polling
/// rather than spin at `DRAIN_POLL_INTERVAL` forever (CONSTITUTION rule 10).
pub(crate) fn drain_pending<T>(rx: &crossbeam_channel::Receiver<T>) -> Option<Vec<T>> {
    let mut events = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(ev) => events.push(ev),
            Err(crossbeam_channel::TryRecvError::Empty) => return Some(events),
            Err(crossbeam_channel::TryRecvError::Disconnected) => return None,
        }
    }
}

/// Compute the hide deadline for a finished `SessionCompleted`.
///
/// Delegates the actual duration to `overlay::outcome_hide_delay` — the
/// single source of truth (CONSTITUTION rule 26) shared with the
/// `OverlayModel`'s own on-screen flash timer, so the window never hides
/// before (or long after) the text it's showing has cleared. Per Fix 4,
/// `Inserted` and `ClipboardOnly` now carry different durations (~1.9s vs
/// ~3.1s including margin) instead of one flat delay; `Failed` is genuine
/// data loss — the clipboard write itself failed, so no copy of the text
/// survives anywhere reachable by the user — and mirrors the
/// non-recoverable-`Error` branch by leaving the overlay shown with no
/// deadline at all.
pub(crate) fn hide_at_for_injection(
    injection: &vuho_domain::InjectionOutcome,
    now: Instant,
) -> Option<Instant> {
    overlay::outcome_hide_delay(injection).map(|delay| now + delay)
}

/// Whether a `SessionCompleted` arriving right now is stale — i.e.
/// `session_active` is not currently tracking a session, so there is
/// nothing this completion could legitimately belong to (either a
/// duplicate, or one that arrived after an `Error` already ended tracking
/// for its session).
///
/// `session_active` is `true` from `SessionStarted` until [`track_session`]
/// clears it — on that same session's own `SessionCompleted`, or on an
/// `Error`. A completion is therefore stale exactly when `session_active`
/// is `false`.
pub(crate) fn session_completed_is_stale(session_active: bool) -> bool {
    !session_active
}

/// One event's effect on `session_active`, and whether it should be applied
/// to the overlay model (`false` = skip as stale). The single place this
/// bookkeeping happens (CONSTITUTION rule 26) — `apply_events` calls this
/// for every event in its batch instead of inlining the decision, which is
/// what let the original bug (finding 6) happen: `session_active` was set
/// on `SessionStarted` and cleared on `Error`, but never cleared on the
/// `SessionCompleted` happy path, so every session's own completion misread
/// `session_active` (still `true` from its own `SessionStarted`) as
/// "another session is active" and was wrongly skipped.
///
/// Only `SessionStarted`/`SessionCompleted`/`Error` affect `session_active`;
/// every other variant passes through unskipped and leaves it untouched.
fn track_session(event: &DictationEvent, session_active: &mut bool) -> bool {
    match event {
        DictationEvent::SessionStarted => {
            *session_active = true;
            true
        }
        DictationEvent::SessionCompleted { .. } => {
            if session_completed_is_stale(*session_active) {
                false
            } else {
                // This session's own completion: stop tracking it now, so a
                // later, spurious second completion (no intervening
                // SessionStarted) is correctly recognized as stale instead
                // of misapplied.
                *session_active = false;
                true
            }
        }
        DictationEvent::Error { .. } => {
            *session_active = false;
            true
        }
        DictationEvent::PartialTranscript { .. } | DictationEvent::Activity { .. } => true,
    }
}

/// Apply one drained batch to the panel's overlay entity: forward non-stale
/// events to the model, prompt for mic access on a mic-permission error, and
/// (re)schedule the hide timer — then, once every event in the batch has
/// been forwarded, apply the one panel-level transition the batch implies
/// (`panel::on_session_started` when `show` — `SessionStarted`, or a
/// show-worthy `Error` — was seen). Returns the updated hide deadline.
///
/// `session_active` persists across calls (like `hide_at`) so a stale
/// `SessionCompleted` — one with no session currently being tracked — can
/// be detected and neither applied to the model nor allowed to schedule a
/// hide that would cut off a newer session's live overlay (see
/// [`track_session`]).
// `status: &StatusHandle` is a genuine pointer-sized `Entity<StatusModel>`
// under production; only the demo build's `StatusHandle = ()` makes a
// by-reference 0-byte type here, so the pedantic pass-by-ref lint only ever
// fires under `--features demo` (see `set_recording`'s matching allow).
#[allow(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn apply_events(
    panel: WindowHandle<PanelRoot>,
    events: Vec<DictationEvent>,
    cx: &mut gpui::AsyncApp,
    mut hide_at: Option<Instant>,
    session_active: &mut bool,
    status: &StatusHandle,
) -> Option<Instant> {
    let mut show = false;
    let mut skip = vec![false; events.len()];
    for (ev, skip) in events.iter().zip(skip.iter_mut()) {
        if !track_session(ev, session_active) {
            log::info!("menu: stale event ignored (event={ev:?})");
            *skip = true;
            continue;
        }
        match ev {
            DictationEvent::SessionStarted => {
                log::info!("menu: Start Listening → Stop Listening (session started)");
                show = true;
                hide_at = None;
                set_recording(status, true, cx);
            }
            DictationEvent::SessionCompleted { injection, .. } => {
                log::info!(
                    "menu: Stop Listening → Start Listening (session completed, injection={injection:?})"
                );
                set_recording(status, false, cx);
                hide_at = hide_at_for_injection(injection, Instant::now());
            }
            DictationEvent::Error {
                recoverable,
                kind,
                message,
            } => {
                log::warn!(
                    "menu: Stop Listening → Start Listening (error={kind:?} recoverable={recoverable})"
                );
                log::error!("vuho: ERROR: {message}");
                set_recording(status, false, cx);
                // Show on error too, not just on SessionStarted: the pipeline
                // now emits SessionStarted only once the stream is actually
                // live, so a failed start reaches the UI with the overlay still
                // hidden. Without this the message would render into a window
                // nobody can see. This is what decouples *overlay visibility*
                // from the *success signal*.
                show = true;
                hide_at = if *recoverable {
                    Some(Instant::now() + overlay::DEFAULT_HIDE_DELAY)
                } else {
                    None
                };
                if matches!(kind, ErrorKind::MicPermissionDenied) {
                    permissions::show_microphone_denied();
                }
            }
            DictationEvent::PartialTranscript { .. } | DictationEvent::Activity { .. } => {}
        }
    }
    let _ = panel.update(cx, |root, _window, cx| {
        root.overlay.update(cx, |model, cx| {
            for (ev, skip) in events.into_iter().zip(skip) {
                if skip {
                    continue;
                }
                model.handle_event(ev);
            }
            cx.notify();
        });
    });
    if show {
        // `on_session_started` takes `&mut App`; this function's own `cx` is
        // `&mut AsyncApp` (a distinct type, no implicit coercion between
        // them) — `AsyncApp::update` is the bridge, handing the closure a
        // real `&mut App` for the duration of the call.
        let _ = cx.update(|cx| panel::on_session_started(panel, cx));
    }
    hide_at
}

/// Hide the panel (if it's presenting the Hud) once the outcome-display
/// deadline has passed.
pub(crate) fn maybe_hide(
    panel: WindowHandle<PanelRoot>,
    hide_at: &mut Option<Instant>,
    cx: &mut gpui::AsyncApp,
) {
    if let Some(t) = *hide_at {
        if Instant::now() >= t {
            *hide_at = None;
            // See `apply_events`'s matching comment: bridge `&mut AsyncApp`
            // to the `&mut App` `hide_if_hud` needs via `AsyncApp::update`.
            let _ = cx.update(|cx| panel::hide_if_hud(panel, cx));
        }
    }
}

/// Drain pipeline (or demo) events into the panel's overlay entity for the
/// process lifetime.
///
/// Exits (with a diagnostic) if `event_rx`'s sender is ever dropped, instead
/// of silently spinning (CONSTITUTION rule 10 — finding 4).
pub(crate) fn spawn_event_drain(
    panel: WindowHandle<PanelRoot>,
    event_rx: crossbeam_channel::Receiver<DictationEvent>,
    status: StatusHandle,
    cx: &mut App,
) {
    cx.spawn(move |cx: &mut gpui::AsyncApp| {
        let mut cx = cx.clone();
        async move {
            let mut hide_at: Option<Instant> = None;
            // Persists across drain batches (like `hide_at`) — see
            // `session_completed_is_stale`.
            let mut session_active = false;
            loop {
                let Some(events) = drain_pending(&event_rx) else {
                    log::info!("spawn_event_drain: event channel disconnected — stopping drain");
                    return;
                };
                if !events.is_empty() {
                    hide_at = apply_events(
                        panel,
                        events,
                        &mut cx,
                        hide_at,
                        &mut session_active,
                        &status,
                    );
                }
                maybe_hide(panel, &mut hide_at, &mut cx);
                cx.background_executor().timer(DRAIN_POLL_INTERVAL).await;
            }
        }
    })
    .detach();
}

/// Drain `UiCommand`s from the status-bar menu's objc2 delegate (which has
/// no GPUI handle of its own) into panel transitions.
///
/// Mirrors [`spawn_event_drain`]'s poll-and-detach shape.
#[cfg(not(feature = "demo"))]
pub(crate) fn spawn_ui_drain(
    panel: WindowHandle<PanelRoot>,
    ui_rx: crossbeam_channel::Receiver<crate::app_state::UiCommand>,
    status: StatusHandle,
    cx: &mut App,
) {
    cx.spawn(move |cx: &mut gpui::AsyncApp| {
        let mut cx = cx.clone();
        async move {
            loop {
                let Some(commands) = drain_pending(&ui_rx) else {
                    log::info!("spawn_ui_drain: ui command channel disconnected — stopping drain");
                    return;
                };
                for command in commands {
                    match command {
                        crate::app_state::UiCommand::OpenPanel => {
                            let _ = cx.update(|cx| panel::open_from_tray(panel, cx));
                        }
                        crate::app_state::UiCommand::ModelStatus(model_status) => {
                            let failed = matches!(
                                model_status,
                                vuho_domain::ModelStatus::Failed { .. }
                            );
                            let _ = status.update(&mut cx, |model, cx| {
                                model.model = Some(model_status.clone());
                                cx.notify();
                            });
                            // Routine Downloading/Verifying ticks must never
                            // surface the panel — only a `Failed` status the
                            // user actually needs to act on, and only if the
                            // panel isn't already open (never steal focus
                            // from whatever the user is doing).
                            if failed {
                                let shown =
                                    cx.update(|cx| panel::is_shown(panel, cx)).unwrap_or(true);
                                if !shown {
                                    let _ = cx.update(|cx| {
                                        panel::show_full(panel, panel::Tab::Settings, cx);
                                    });
                                }
                            }
                        }
                        crate::app_state::UiCommand::EngineReady(Ok(())) => {
                            let _ = status.update(&mut cx, |model, cx| {
                                model.engine = crate::app_status::EngineState::Ready;
                                cx.notify();
                            });
                        }
                        crate::app_state::UiCommand::EngineReady(Err(e)) => {
                            log::error!("vuho: engine warmup failed: {e}");
                            let _ = status.update(&mut cx, |model, cx| {
                                model.engine = crate::app_status::EngineState::Failed(e.clone());
                                cx.notify();
                            });
                        }
                    }
                }
                cx.background_executor().timer(DRAIN_POLL_INTERVAL).await;
            }
        }
    })
    .detach();
}

#[cfg(test)]
mod tests {
    use super::*;
    use vuho_domain::InjectionOutcome;

    // ── Bug 2 / finding 6 fix: stale `SessionCompleted` detection ───────────

    #[test]
    fn session_completed_is_stale_when_no_session_is_currently_tracked() {
        // Nothing is being tracked (already cleared by an Error, or by a
        // prior completion) — a SessionCompleted arriving now has nothing
        // legitimate to apply to.
        assert!(session_completed_is_stale(false));
    }

    #[test]
    fn session_completed_is_not_stale_while_a_session_is_tracked() {
        // A session is being tracked (SessionStarted fired, nothing has
        // cleared it yet) — this is that session's own, legitimate
        // completion.
        assert!(!session_completed_is_stale(true));
    }

    /// The literal scenario finding 6 named: `SessionStarted` →
    /// `SessionCompleted` must apply (not skip) the completion, and a
    /// second `SessionCompleted` with no intervening `SessionStarted` must
    /// be treated as stale. This is `apply_events`' actual bookkeeping
    /// chokepoint — no GPUI window/context needed, since `track_session`
    /// is the pure decision `apply_events` delegates to.
    #[test]
    fn track_session_applies_the_owning_completion_and_flags_a_second_one_stale() {
        let mut session_active = false;

        assert!(track_session(
            &DictationEvent::SessionStarted,
            &mut session_active
        ));
        assert!(session_active);

        let completed = DictationEvent::SessionCompleted {
            result: vuho_domain::TranscriptionResult {
                segments: vec![],
                full_text: "hello".to_string(),
                language: "en".to_string(),
            },
            injection: InjectionOutcome::Inserted,
        };

        assert!(
            track_session(&completed, &mut session_active),
            "a session's own completion must be applied, not skipped as stale"
        );
        assert!(
            !session_active,
            "session_active must be cleared once this session's completion is applied \
             (finding 6 — the original bug never cleared it)"
        );

        assert!(
            !track_session(&completed, &mut session_active),
            "a second SessionCompleted with no intervening SessionStarted must be treated as stale"
        );
    }

    #[test]
    fn hide_at_for_injection_inserted_schedules_hide() {
        // Fix 4: per-outcome duration, no longer a single flat delay —
        // exercised via `overlay::outcome_hide_delay`, the shared source of
        // truth this function now delegates to entirely.
        let now = Instant::now();
        let hide_at = hide_at_for_injection(&InjectionOutcome::Inserted, now);
        assert_eq!(
            hide_at,
            overlay::outcome_hide_delay(&InjectionOutcome::Inserted).map(|d| now + d)
        );
        assert!(hide_at.is_some());
    }

    #[test]
    fn hide_at_for_injection_clipboard_only_outlasts_inserted() {
        // Fix 4: the clipboard-fallback note carries an instruction the user
        // must act on, so it stays up noticeably longer than a bare
        // confirmation — both auto-hide, but ClipboardOnly's deadline is later.
        let now = Instant::now();
        let inserted_at = hide_at_for_injection(&InjectionOutcome::Inserted, now).unwrap();
        let clipboard_at = hide_at_for_injection(
            &InjectionOutcome::ClipboardOnly {
                reason: "secure input active".into(),
            },
            now,
        )
        .unwrap();
        assert!(
            clipboard_at > inserted_at,
            "ClipboardOnly ({clipboard_at:?}) must outlast Inserted ({inserted_at:?})"
        );
    }

    #[test]
    fn hide_at_for_injection_failed_stays_visible() {
        // Genuine data loss: no hide deadline, mirroring the non-recoverable
        // `Error` branch — the overlay stays up until the next session.
        let now = Instant::now();
        let hide_at = hide_at_for_injection(
            &InjectionOutcome::Failed {
                reason: "clipboard write failed".into(),
            },
            now,
        );
        assert_eq!(hide_at, None);
    }
}
