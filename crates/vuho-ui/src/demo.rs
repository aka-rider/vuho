//! Demo mode: feeds synthetic `DictationEvent`s into the panel's overlay
//! entity so `cargo run -p vuho-ui --features demo` previews the UI with no
//! microphone or engine required. Split out of `main.rs` (WP10) — this
//! whole module is `#[cfg(feature = "demo")]`. The demo panel never leaves
//! the Hud presentation (see `panel.rs`'s module doc comment), so this
//! drives events into `panel`'s overlay entity exactly as the old
//! standalone overlay window did.

use std::time::Duration;

use gpui::{App, WindowHandle};
use vuho_domain::DictationEvent;

use crate::event_loop::spawn_event_drain;
use crate::panel::PanelRoot;

/// Duration for demo session phases. Only used by `run_demo_mode`.
const DEMO_PAUSE_DURATION: Duration = Duration::from_secs(2);
const DEMO_UPDATE_INTERVAL: Duration = Duration::from_millis(800);

/// Demo phrases: `(full_accumulated_text, unconfirmed_suffix)` — see
/// `run_demo_session`'s doc comment for how `confirmed_text` is derived
/// from this. Short script, always completes with
/// `InjectionOutcome::Inserted` — see [`DEMO_PHRASES_LONG`]
/// for the wrapping/`ClipboardOnly` counterpart.
const DEMO_PHRASES: &[(&str, &str)] = &[
    ("And so my", ""),
    ("And so my dear", " dear"),
    ("And so my dear friends", " friends"),
    ("And so my dear friends, we must carry on", " carry on"),
    (
        "And so my dear friends, we must carry on with courage",
        " with courage",
    ),
];

/// A longer demo script (Fix 3/4 follow-up): the growing paragraph overflows
/// the 3-line transcript viewport well before it finishes, exercising the
/// multi-line wrap and top fade; its `SessionCompleted` reports
/// `ClipboardOnly` (see [`demo_script`]) to exercise Fix 4's other outcome
/// wording/duration, which [`DEMO_PHRASES`] alone never demonstrates.
const DEMO_PHRASES_LONG: &[(&str, &str)] = &[
    ("Four score and seven years ago", ""),
    (
        "Four score and seven years ago our fathers brought forth on this continent",
        " brought forth on this continent",
    ),
    (
        "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty",
        " a new nation, conceived in liberty",
    ),
    (
        "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal.",
        " and dedicated to the proposition that all men are created equal.",
    ),
];

/// Start the demo mode: feed synthetic events into the panel's overlay
/// entity.
///
/// Uses the same `spawn_event_drain` the production wiring uses, so the demo
/// previews the real lifecycle (outcome flash + delayed hide). Alternates
/// between [`DEMO_PHRASES`] (short, `Inserted`) and [`DEMO_PHRASES_LONG`]
/// (long, `ClipboardOnly`) every cycle — see [`demo_script`] — so a demo run
/// previews every outcome wording/duration (Fix 4) and the multi-line
/// wrap/fade (Fix 3), not just the original single short/`Inserted` path.
pub(crate) fn run_demo_mode(panel: WindowHandle<PanelRoot>, cx: &mut App) {
    let (demo_tx, demo_rx) = crossbeam_channel::unbounded();

    // No `StatusModel` in demo mode (no menu bar, no settings) — `()` is
    // `event_loop::StatusHandle`'s demo-build value.
    spawn_event_drain(panel, demo_rx, (), cx);

    cx.spawn(move |cx: &mut gpui::AsyncApp| {
        let cx = cx.clone();
        async move {
            let mut cycle: usize = 0;
            loop {
                let (phrases, injection) = demo_script(cycle);
                run_demo_session(&demo_tx, phrases, injection, &cx).await;
                cycle = cycle.wrapping_add(1);
                cx.background_executor().timer(DEMO_PAUSE_DURATION).await;
            }
        }
    })
    .detach();
}

/// The demo script for a given cycle index: which phrases to stream, and
/// the `InjectionOutcome` its `SessionCompleted` reports. Pure and
/// deterministic (`cycle.is_multiple_of(2)`) so the alternation is easy to reason about
/// and to unit-test.
fn demo_script(
    cycle: usize,
) -> (
    &'static [(&'static str, &'static str)],
    vuho_domain::InjectionOutcome,
) {
    use vuho_domain::InjectionOutcome;

    if cycle.is_multiple_of(2) {
        (DEMO_PHRASES, InjectionOutcome::Inserted)
    } else {
        (
            DEMO_PHRASES_LONG,
            InjectionOutcome::ClipboardOnly {
                reason: "demo: secure input active".into(),
            },
        )
    }
}

/// Stream one demo session's `phrases` into `demo_tx` as growing
/// `PartialTranscript`s (`DEMO_UPDATE_INTERVAL` apart), then
/// `SessionCompleted` with `injection`. The completed transcript's
/// `full_text` is the last phrase's full accumulated text — never
/// re-derived or hardcoded separately (CONSTITUTION rule 26).
///
/// `phrases` stores `(full_accumulated_text, unconfirmed_suffix)` — the
/// same hand-authored shape as before the WP6 `confirmed_text` domain
/// change, kept because it reads naturally as a growing script. The
/// producer-supplied `confirmed_text` this function actually sends is
/// derived from it once, right here, at demo-data-authoring time — this is
/// NOT the UI re-deriving a fact from a live domain event (the rule-2
/// anti-pattern `overlay.rs`'s retired `split_transcript` was): it is this
/// module *acting as* the producer, synthesizing fake `DictationEvent`s the
/// same way `vuho-stt-engine`'s real `Accumulator` would, from literal
/// strings it already owns outright.
async fn run_demo_session(
    demo_tx: &crossbeam_channel::Sender<DictationEvent>,
    phrases: &[(&str, &str)],
    injection: vuho_domain::InjectionOutcome,
    cx: &gpui::AsyncApp,
) {
    use vuho_domain::TranscriptionResult;

    let _ = demo_tx.send(DictationEvent::SessionStarted);

    let mut full_text = String::new();
    for (accumulated_text, unconfirmed_text) in phrases {
        full_text = (*accumulated_text).to_string();
        let confirmed_text = accumulated_text
            .strip_suffix(unconfirmed_text)
            .unwrap_or(accumulated_text)
            .to_string();
        let _ = demo_tx.send(DictationEvent::PartialTranscript {
            confirmed_text,
            unconfirmed_text: (*unconfirmed_text).to_string(),
        });
        let _ = demo_tx.send(DictationEvent::Activity { level: 0.8 });
        cx.background_executor().timer(DEMO_UPDATE_INTERVAL).await;
    }

    let _ = demo_tx.send(DictationEvent::SessionCompleted {
        result: TranscriptionResult {
            segments: vec![],
            full_text,
            language: "en".to_string(),
        },
        injection,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use vuho_domain::InjectionOutcome;

    #[test]
    fn demo_script_alternates_phrases_and_outcome() {
        // Compared by content, not `.as_ptr()`: `const` slices don't
        // guarantee a stable address across use sites in Rust, so pointer
        // identity is not a valid way to tell "which script" this is.
        let (phrases_even, injection_even) = demo_script(0);
        assert_eq!(phrases_even, DEMO_PHRASES);
        assert!(matches!(injection_even, InjectionOutcome::Inserted));

        let (phrases_odd, injection_odd) = demo_script(1);
        assert_eq!(phrases_odd, DEMO_PHRASES_LONG);
        assert!(matches!(
            injection_odd,
            InjectionOutcome::ClipboardOnly { .. }
        ));

        // Deterministic on cycle parity alone, not on call order.
        let (phrases_even_again, _) = demo_script(42);
        assert_eq!(phrases_even_again, DEMO_PHRASES);
    }

    #[test]
    fn demo_phrases_long_exceeds_three_lines_worth_of_text() {
        // Fix 3 follow-up: the long script's final phrase must actually be
        // long enough to overflow the 3-line transcript viewport at 16px
        // text — otherwise the demo never exercises the wrap/fade it exists
        // to preview. ~30 chars/line at 16px in a 460px-wide panel is a
        // conservative estimate; 3 lines' worth is ~90 chars.
        let (final_text, _) = DEMO_PHRASES_LONG
            .last()
            .expect("DEMO_PHRASES_LONG must be non-empty");
        assert!(
            final_text.len() > 90,
            "final phrase too short to demo wrapping: {final_text:?}"
        );
    }
}
