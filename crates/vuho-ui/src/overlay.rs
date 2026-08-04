//! Overlay view model: transcript, recording LED, waveform state + render.

use gpui::{
    div, hsla, linear_color_stop, linear_gradient, point, prelude::*, px, Animation, AnimationExt,
    AnyElement, AsyncApp, BoxShadow, Context, HighlightStyle, Hsla, Pixels, SharedString,
    StyledText, WeakEntity, Window,
};
use std::time::{Duration, Instant};
use vuho_domain::{DictationEvent, InjectionOutcome};

use crate::theme;

/// Number of waveform bars.
const BAR_COUNT: usize = 12;

/// Animation frame interval (~30 fps).
const ANIMATION_INTERVAL: std::time::Duration = std::time::Duration::from_millis(33);

// ── Outcome on-screen durations (Fix 4) ─────────────────────────────────────
//
// Each `SessionCompleted` outcome flashes for a duration proportional to how
// much the user needs to read: a bare confirmation is quick, but the
// clipboard fallback carries an instruction the user must act on, so it
// stays up noticeably longer. `None` (see `Outcome::display_duration`,
// `outcome_hide_delay`) means "never auto-clear" — reserved for genuine data
// loss (`InjectionOutcome::Failed`, non-recoverable `Error`s), which must
// stay until the next session rather than flash away unread.

// Raw millisecond magnitudes, kept alongside the `Duration` constants below
// so `DEFAULT_HIDE_DELAY` can derive from them in a `const` context (`Add`
// on `Duration` isn't `const fn`; plain integer addition is).
const INSERTED_DISPLAY_MS: u64 = 1800;
const CLIPBOARD_ONLY_DISPLAY_MS: u64 = 3000;
const DEFAULT_MESSAGE_DISPLAY_MS: u64 = 800;
const HIDE_MARGIN_MS: u64 = 100;

/// How long "✓ Inserted" stays on screen.
const INSERTED_DISPLAY: Duration = Duration::from_millis(INSERTED_DISPLAY_MS);
/// How long the clipboard-fallback note stays on screen — longer than
/// `INSERTED_DISPLAY` because it asks the user to act (press ⌘V themselves).
const CLIPBOARD_ONLY_DISPLAY: Duration = Duration::from_millis(CLIPBOARD_ONLY_DISPLAY_MS);
/// Default duration for a plain recoverable message (session errors surfaced
/// via `Outcome::Message`) — the original single-constant behavior, kept for
/// the one outcome kind Fix 4 didn't ask to change.
const DEFAULT_MESSAGE_DISPLAY: Duration = Duration::from_millis(DEFAULT_MESSAGE_DISPLAY_MS);
/// Extra margin `main.rs`'s drain task keeps the window visible after an
/// outcome's own on-screen duration elapses, so the flash is fully seen
/// before the window itself hides.
const HIDE_MARGIN: Duration = Duration::from_millis(HIDE_MARGIN_MS);

/// `main.rs`'s hide delay for the one case that isn't `SessionCompleted`
/// (a recoverable `DictationEvent::Error`) — unaffected by Fix 4, so it
/// keeps deriving from the original default duration.
pub(crate) const DEFAULT_HIDE_DELAY: Duration =
    Duration::from_millis(DEFAULT_MESSAGE_DISPLAY_MS + HIDE_MARGIN_MS);

/// How long `main.rs` should keep the overlay window visible after a
/// `SessionCompleted` with this injection outcome, or `None` to leave it
/// showing indefinitely. The single source of truth `main.rs` calls instead
/// of re-deriving a duration from `InjectionOutcome` itself (CONSTITUTION
/// rule 26) — mirrors `Outcome::display_duration` below, which drives the
/// same flash's on-screen clock inside `OverlayModel`.
#[must_use]
pub(crate) fn outcome_hide_delay(injection: &InjectionOutcome) -> Option<Duration> {
    let display = match injection {
        InjectionOutcome::Inserted => INSERTED_DISPLAY,
        InjectionOutcome::ClipboardOnly { .. } => CLIPBOARD_ONLY_DISPLAY,
        // Nothing was delivered anywhere, deliberately (blank transcript):
        // a brief note is enough — there is no instruction to act on and
        // no data at risk.
        InjectionOutcome::NothingToInject => DEFAULT_MESSAGE_DISPLAY,
        // Genuine data loss: never auto-hide, since there's no known-safe
        // urgency to assume for a failed clipboard write.
        InjectionOutcome::Failed { .. } => return None,
    };
    Some(display + HIDE_MARGIN)
}

/// Text size for the transcript / outcome message rows: `theme::TEXT_LG`.
const TRANSCRIPT_TEXT_SIZE: Pixels = px(theme::TEXT_LG);
/// Line height for the wrapped transcript paragraph (Fix 3: ~1.4x text size).
const TRANSCRIPT_LINE_HEIGHT: Pixels = px(22.0);
/// Visible transcript viewport height: 3 lines at `TRANSCRIPT_LINE_HEIGHT`.
/// Older lines scroll out the top, masked by `fade_strip` instead of a hard
/// clip (Fix 3's "bottom-anchored, newest text always visible").
const TRANSCRIPT_HEIGHT: Pixels = px(66.0);
/// Height of the fade-out gradient strip over the top of the transcript.
const FADE_HEIGHT: Pixels = px(18.0);

// ── Palette ──────────────────────────────────────────────────────────────
//
// Small, coherent set of named saturation/lightness/opacity magnitudes
// shared by the color helpers below (CONSTITUTION rule 27 — name semantic
// numbers, not just the hues). Design principle (user correction after
// review): separate elements with text color — opacity/contrast and
// typography — first; hue is a last resort. The only saturated hue left on
// the panel is the recording LED (`theme::ERROR_RED`); everything else —
// transcript, outcome text, the ⌘V chip, the waveform — is neutral
// white/gray, distinguished by opacity and size alone.
//
// The prominent/secondary/disabled text opacities and the recording red
// itself now live in `theme.rs` (the crate's shared visual-language
// chokepoint) as `TEXT_PRIMARY`/`TEXT_SECONDARY`/`TEXT_DISABLED`/
// `ERROR_RED`; only the magnitudes with no cross-window equivalent — panel
// chrome, the waveform's own faint alphas, the idle recording dot, the LED
// pulse/glow geometry — stay local to this file.

/// Neutral (gray/white) hue: 0 with zero saturation.
const NEUTRAL_HUE: f32 = 0.0;
const NEUTRAL_SATURATION: f32 = 0.0;

/// Full white, used for the waveform, panel border, and key-cap chip border.
const LIGHTNESS_WHITE: f32 = 1.0;
/// Mid gray, used for the idle recording dot.
const LIGHTNESS_IDLE_DOT: f32 = 0.5;

/// Opacity of the active recording dot, applied over `theme::ERROR_RED`.
const OPACITY_ACTIVE_DOT: f32 = 0.9;
/// Waveform bars are ambient activity texture, not information — a faint
/// alpha pulse is enough; no hue shift with recording state (that's the red
/// LED's job alone, Fix 5).
const WAVEFORM_ACTIVE_ALPHA: f32 = 0.3;
const WAVEFORM_IDLE_ALPHA: f32 = 0.12;
/// Opacity of the idle recording dot — same magnitude as
/// `theme::TEXT_DISABLED`'s alpha, deliberately de-emphasized relative to
/// the active state.
const OPACITY_DIMMED: f32 = 0.4;

// Panel chrome (Fix 3: moved out of the inline literals at the old
// `overlay.rs:357-359` and into this named block, so the fade strip below
// can reference the exact same background color). The hue/saturation/
// lightness themselves live in `theme::PANEL_HUE`/`PANEL_SATURATION`/
// `PANEL_LIGHTNESS` (F20) — shared with `panel.rs`'s `FULL_BG`; only the
// opacity below is unique to the overlay's floating, semi-transparent panel.
/// Raised from 0.85 (Fix 3): text should never fight the desktop behind it.
const PANEL_BG_OPACITY: f32 = 0.9;
const PANEL_BORDER_OPACITY: f32 = 0.1;

// Recording LED (Fix 5): warm red (`theme::ERROR_RED`), replacing the old
// green/gray dot — the only saturated hue on the panel; every other element
// (transcript, outcome text, chip, waveform) is neutral white/gray (see the
// palette note above).
const OPACITY_GLOW: f32 = 0.5;
const LED_SIZE: Pixels = px(8.0);
/// Distance from the panel's top-left corner to the LED — freed up from the
/// old bottom row, where a status light doesn't read naturally.
const LED_INSET: Pixels = px(10.0);
/// One breathing cycle: slow enough to read as "alive", not a blink.
const LED_PULSE_DURATION: Duration = Duration::from_millis(1600);
const LED_PULSE_MIN: f32 = 0.55;
const LED_PULSE_MAX: f32 = 1.0;
const LED_GLOW_BLUR: Pixels = px(6.0);
const LED_GLOW_SPREAD: Pixels = px(1.0);

// Waveform: ambient texture, recedes behind the transcript (user correction
// after review) — smaller max bar height (was 24px) and breathing room
// (`WAVEFORM_TOP_MARGIN`) above it so it doesn't crowd the text.
//
// `WAVEFORM_MAX_BAR_HEIGHT` is the raw magnitude (not `Pixels`, whose inner
// f32 is crate-private to gpui) that `render_waveform` scales each bar's
// amplitude by; `WAVEFORM_HEIGHT` derives from it so the container and the
// tallest possible bar can never drift apart (CONSTITUTION rule 26).
const WAVEFORM_MAX_BAR_HEIGHT: f32 = 16.0;
const WAVEFORM_HEIGHT: Pixels = px(WAVEFORM_MAX_BAR_HEIGHT);
const WAVEFORM_BAR_WIDTH: Pixels = px(3.0);
const WAVEFORM_BAR_MIN_HEIGHT: Pixels = px(2.0);
const WAVEFORM_BAR_RADIUS: Pixels = px(1.5);
const WAVEFORM_TOP_MARGIN: Pixels = px(8.0);

// Key-cap chip (Fix 4). Radius is `theme::RADIUS_CHIP`, applied at the
// chip's `.rounded()` call site; text size is `theme::TEXT_SM`.
const KEYCAP_BORDER_OPACITY: f32 = 0.25;
const KEYCAP_TEXT_SIZE: Pixels = px(theme::TEXT_SM);
/// Chip + hint opacity for the dimmed ("to paste again") vs. prominent
/// ("to paste") wordings — see `Outcome::chip_hint`.
const CHIP_OPACITY_DIMMED: f32 = 0.6;
const CHIP_OPACITY_PROMINENT: f32 = 0.95;

fn color_message() -> Hsla {
    theme::TEXT_SECONDARY
}

/// Neutral white, alpha-only distinction between active/idle — no hue shift
/// with recording state (that's the LED's job alone; see the palette note).
fn color_waveform_bar(recording: bool) -> Hsla {
    hsla(
        NEUTRAL_HUE,
        NEUTRAL_SATURATION,
        LIGHTNESS_WHITE,
        if recording {
            WAVEFORM_ACTIVE_ALPHA
        } else {
            WAVEFORM_IDLE_ALPHA
        },
    )
}

/// Recording LED color: warm red (`theme::ERROR_RED`) while recording,
/// neutral gray idle (unchanged idle behavior — only the active color moved
/// off green).
fn color_recording_dot(recording: bool) -> Hsla {
    if recording {
        theme::ERROR_RED.opacity(OPACITY_ACTIVE_DOT)
    } else {
        hsla(
            NEUTRAL_HUE,
            NEUTRAL_SATURATION,
            LIGHTNESS_IDLE_DOT,
            OPACITY_DIMMED,
        )
    }
}

fn color_recording_glow() -> Hsla {
    theme::ERROR_RED.opacity(OPACITY_GLOW)
}

/// Confirmed transcript text: `theme::TEXT_PRIMARY`.
fn color_confirmed_text() -> Hsla {
    theme::TEXT_PRIMARY
}

/// Unconfirmed (in-flight) transcript tail: `theme::TEXT_DISABLED` —
/// visually distinguishes text the engine hasn't confirmed yet.
fn color_unconfirmed_text() -> Hsla {
    theme::TEXT_DISABLED
}

fn color_panel_bg() -> Hsla {
    hsla(
        theme::PANEL_HUE,
        theme::PANEL_SATURATION,
        theme::PANEL_LIGHTNESS,
        PANEL_BG_OPACITY,
    )
}

fn color_panel_border() -> Hsla {
    hsla(
        NEUTRAL_HUE,
        NEUTRAL_SATURATION,
        LIGHTNESS_WHITE,
        PANEL_BORDER_OPACITY,
    )
}

/// The outcome of a finished session, shown briefly in place of the transcript.
#[derive(Clone)]
enum Outcome {
    /// Text was injected into the focused app.
    Inserted,
    /// The pipeline fell back to clipboard-only (e.g. Secure Input active) —
    /// the text is safely on the clipboard, but ⌘V was not sent for the
    /// user, so the hint above is worded as an instruction, not a recap.
    ClipboardOnly,
    /// A message to surface (e.g. a session error, or the `Failed` injection
    /// outcome — a genuine clipboard-write failure).
    ///
    /// `persistent == true` is not auto-cleared by a timer — it stays until
    /// the next `SessionStarted`.
    Message {
        text: SharedString,
        persistent: bool,
    },
}

impl Outcome {
    /// How long this outcome stays on screen before `OverlayModel` clears it.
    /// `None` means never auto-clear. Mirrors `outcome_hide_delay` above,
    /// which drives `main.rs`'s window-hide timer from the same constants.
    fn display_duration(&self) -> Option<Duration> {
        match self {
            Outcome::Inserted => Some(INSERTED_DISPLAY),
            Outcome::ClipboardOnly => Some(CLIPBOARD_ONLY_DISPLAY),
            Outcome::Message {
                persistent: true, ..
            } => None,
            Outcome::Message {
                persistent: false, ..
            } => Some(DEFAULT_MESSAGE_DISPLAY),
        }
    }

    /// The key-cap hint wording + prominence for this outcome (Fix 4).
    /// `None` for `Processing`/`Message` — never show a false clipboard claim
    /// next to a still-running cleanup, a failed clipboard write, or a
    /// session error.
    fn chip_hint(&self) -> Option<(&'static str, bool)> {
        match self {
            Outcome::Inserted => Some(("to paste again", false)),
            Outcome::ClipboardOnly => Some(("to paste", true)),
            Outcome::Message { .. } => None,
        }
    }
}

/// Overlay model state.
///
/// `amplitudes`/`last_nudge` are plain fields, not `Mutex`-wrapped (WP10):
/// `OverlayModel` is a main-thread GPUI `Entity`, and every method here that
/// touches them takes `&mut self` — GPUI's own update mechanism already
/// guarantees exclusive access for the duration of a `Context<Self>`
/// borrow, so the `Mutex` was never protecting against real concurrent
/// access, only adding `.lock().unwrap()` panic sites for no safety benefit.
pub(crate) struct OverlayModel {
    recording: bool,
    confirmed_text: SharedString,
    unconfirmed_text: SharedString,
    amplitudes: Vec<f32>,
    phases: [f32; BAR_COUNT],
    /// Liveliness value set to 1.0 on each activity tick.
    /// Decays visually in the waveform based on elapsed time.
    activity_tick: f32,
    /// Timestamp of the last activity tick.
    last_nudge: Instant,
    /// Transient end-of-session confirmation, cleared after its
    /// `Outcome::display_duration`.
    outcome: Option<Outcome>,
    /// When `outcome` was set (used to expire it in the animation loop).
    outcome_since: Instant,
    /// When this model was constructed — the epoch `update_amplitudes`'s
    /// sine wave phase advances from (WP10: fixes the frozen-waveform bug,
    /// rule 22). The bug: the old code computed `now = Instant::now()` and
    /// then immediately read `now.elapsed()` in the same call, which is
    /// always ≈0 — the sine term was therefore a *constant* per bar (only
    /// `phases[i]` varied, never true elapsed time), so the waveform never
    /// actually animated; only the small random `jitter` term changed
    /// frame to frame. Using `created_at.elapsed()` instead gives a value
    /// that genuinely grows across calls.
    created_at: Instant,
}

impl OverlayModel {
    pub(crate) fn new(_window: &mut Window, cx: &mut Context<Self>) -> Self {
        // SAFETY: BAR_COUNT is 12, so i is at most 11 — safely representable in f32.
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let phases: [f32; BAR_COUNT] = std::array::from_fn(|i| (i as f32) * 0.5);

        let mut model = Self {
            recording: false,
            confirmed_text: SharedString::default(),
            unconfirmed_text: SharedString::default(),
            amplitudes: vec![0.0; BAR_COUNT],
            phases,
            activity_tick: 0.0,
            last_nudge: Instant::now(),
            outcome: None,
            outcome_since: Instant::now(),
            created_at: Instant::now(),
        };

        // Start the waveform animation loop.
        model.start_animation(cx);

        model
    }

    /// Start the periodic waveform animation task.
    ///
    /// Exits (with a diagnostic) once `this` (a `WeakEntity`) can no longer
    /// be upgraded — i.e. the overlay window/model has been dropped —
    /// instead of polling a dead entity forever (WP10: symmetric with
    /// `event_loop`'s drains, which already exit on their channel's
    /// disconnect rather than spinning; CONSTITUTION rule 10).
    #[allow(clippy::unused_self)]
    fn start_animation(&mut self, cx: &mut Context<Self>) {
        cx.spawn(|this: WeakEntity<OverlayModel>, cx: &mut AsyncApp| {
            let mut cx = cx.clone();
            async move {
                loop {
                    cx.background_executor().timer(ANIMATION_INTERVAL).await;
                    let updated = this.update(&mut cx, |model, cx| {
                        model.update_amplitudes();
                        cx.notify();
                    });
                    if updated.is_err() {
                        log::info!("overlay: animation loop stopping — model entity gone");
                        return;
                    }
                }
            }
        })
        .detach();
    }

    /// Update waveform bar amplitudes based on liveliness decay + sine jitter.
    fn update_amplitudes(&mut self) {
        // Expire the end-of-session confirmation.
        let expired = self
            .outcome
            .as_ref()
            .and_then(Outcome::display_duration)
            .is_some_and(|d| self.outcome_since.elapsed() >= d);
        if expired {
            self.outcome = None;
        }

        let elapsed = self.last_nudge.elapsed().as_secs_f32();

        // Exponential decay: liveliness drops by ~5% per 100ms.
        let liveliness = self.activity_tick * 0.95_f32.powf(elapsed * 10.0);
        let baseline = 0.05; // minimum liveliness for ambient pulse

        // Precision loss is acceptable: we only need rough phase for the sine wave
        // animation, not exact timestamps.
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let elapsed_ms = self.created_at.elapsed().as_millis() as f32;

        for i in 0..BAR_COUNT {
            let phase = self.phases[i];
            let base = wave_base(elapsed_ms, phase);
            // Jitter scales with liveliness — more noise when active.
            let jitter = fastrand::f32() * 0.1 * liveliness;
            // Combine: baseline pulse + liveliness-scaled animation + jitter.
            let amp = (baseline + base * liveliness + jitter).clamp(0.02, 1.0);
            self.amplitudes[i] = amp;
        }
    }

    pub(crate) fn handle_event(&mut self, event: DictationEvent) {
        match event {
            DictationEvent::SessionStarted => {
                self.recording = true;
                self.confirmed_text = SharedString::default();
                self.unconfirmed_text = SharedString::default();
                self.outcome = None;
                self.nudge_activity();
            }
            DictationEvent::PartialTranscript {
                confirmed_text,
                unconfirmed_text,
            } => {
                // Both fields are producer-supplied (vuho-stt-engine's
                // Accumulator owns the confirmed/unconfirmed split) — no
                // slicing or re-derivation here, unlike the retired
                // suffix-subtraction reconstruction this replaced (ADR-018).
                self.confirmed_text = SharedString::from(confirmed_text);
                self.unconfirmed_text = SharedString::from(unconfirmed_text);
                self.nudge_activity();
            }
            DictationEvent::Activity { level } => {
                self.activity_tick = level.clamp(0.0, 1.0);
                self.nudge_activity();
            }
            DictationEvent::SessionCompleted { injection, .. } => {
                self.recording = false;
                self.activity_tick = 0.0;
                self.confirmed_text = SharedString::default();
                self.unconfirmed_text = SharedString::default();
                self.set_outcome(outcome_for_injection(&injection));
            }
            DictationEvent::Error {
                message,
                recoverable,
                ..
            } => {
                self.recording = false;
                self.activity_tick = 0.0;
                self.confirmed_text = SharedString::default();
                self.unconfirmed_text = SharedString::default();
                self.set_outcome(Outcome::Message {
                    text: message.into(),
                    persistent: !recoverable,
                });
            }
        }
    }

    fn nudge_activity(&mut self) {
        self.activity_tick = 1.0;
        self.last_nudge = Instant::now();
    }

    /// Set the transient end-of-session confirmation and stamp its time.
    fn set_outcome(&mut self, outcome: Outcome) {
        self.outcome = Some(outcome);
        self.outcome_since = Instant::now();
    }

    /// Compute the liveliness value after a given elapsed time,
    /// matching the decay formula in `update_amplitudes`.
    ///
    /// Test-only: exists so unit tests can verify the decay curve without
    /// spawning a full GPUI context.
    #[cfg(test)]
    pub(crate) fn compute_liveliness(activity_tick: f32, elapsed_secs: f32) -> f32 {
        let liveliness = activity_tick * 0.95_f32.powf(elapsed_secs * 10.0);
        let baseline = 0.05;
        (baseline + liveliness).clamp(0.02, 1.0)
    }
}

/// One waveform bar's sine-wave base value (before jitter), in `[0.0, 1.0]`.
///
/// Pure and unit-tested with directly-supplied `elapsed_ms` values (no
/// wall-clock sleep needed, per CONSTITUTION rule 32) — `elapsed_ms` must be
/// genuine elapsed time since a fixed epoch (`OverlayModel::created_at`,
/// not a freshly-taken `Instant::now()`) for the waveform to actually
/// animate; see [`OverlayModel::created_at`]'s doc comment for the bug this
/// fixes. `bar_phase` is each bar's own fixed offset, giving bars an
/// organic, non-synchronized appearance.
fn wave_base(elapsed_ms: f32, bar_phase: f32) -> f32 {
    (elapsed_ms * 0.003 + bar_phase).sin() * 0.5 + 0.5
}

/// Map a finished session's `InjectionOutcome` to the transient on-screen
/// `Outcome` — the one place this mapping happens (CONSTITUTION rule 26).
/// The pipeline's `reason` strings are already logged by `event_loop.rs`'s
/// `log::info!` on every `SessionCompleted`, so dropping them from the
/// on-screen `ClipboardOnly` text (Fix 4's fixed "Copied to clipboard"
/// wording) loses no diagnostic information.
fn outcome_for_injection(injection: &InjectionOutcome) -> Outcome {
    match injection {
        InjectionOutcome::Inserted => Outcome::Inserted,
        InjectionOutcome::ClipboardOnly { .. } => Outcome::ClipboardOnly,
        // Genuine data loss: the clipboard write itself failed, so no copy
        // of the text survives anywhere reachable by the user. Stays on
        // screen until the next `SessionStarted` instead of auto-clearing.
        InjectionOutcome::Failed { reason } => Outcome::Message {
            text: reason.clone().into(),
            persistent: true,
        },
        // A blank transcript: injection was deliberately skipped and the
        // clipboard left untouched, so no ⌘V chip and no success claim —
        // just a brief auto-clearing note (rule 11: tell the truth).
        InjectionOutcome::NothingToInject => Outcome::Message {
            text: "Nothing to insert".into(),
            persistent: false,
        },
    }
}

/// Wrap `content` in the Hud presentation's floating, translucent panel
/// chrome — background/border/rounded/shadow, sized to fill the window.
/// The one place this exact chrome is built (`panel.rs`'s Hud arm is its
/// only caller); keeping it here rather than in `panel.rs` lets
/// `color_panel_bg`/`color_panel_border` stay private to this module.
/// `panel.rs`'s Full presentation builds its own (opaque) chrome instead —
/// the Overlay tab embeds [`OverlayModel::render_content`] directly, with no
/// second background of its own (see that method's doc comment).
pub(crate) fn hud_chrome(content: AnyElement) -> AnyElement {
    div()
        .relative()
        .size_full()
        .bg(color_panel_bg())
        .border_1()
        .border_color(color_panel_border())
        .rounded(px(theme::RADIUS_PANEL))
        .shadow_lg()
        .child(content)
        .into_any_element()
}

impl OverlayModel {
    /// The overlay's live content: the transcript/outcome line, the ambient
    /// waveform, and the recording LED — no outer chrome (background/
    /// border/shadow) of its own. [`hud_chrome`] wraps this for the Hud
    /// presentation; the Full presentation's Overlay tab (`panel.rs`) embeds
    /// it directly inside the panel's own opaque chrome, so this must never
    /// paint a second background — that's why the panel-sizing/background
    /// styling that used to live in `impl Render for OverlayModel` stays out
    /// of this method.
    pub(crate) fn render_content(&self) -> AnyElement {
        let amps = &self.amplitudes;
        div()
            .relative()
            .flex()
            .flex_col()
            .gap_2()
            .px_6()
            .py_4()
            .child(self.render_transcript_area())
            .child(render_waveform(amps, self.recording))
            .child(render_recording_led(self.recording))
            .into_any_element()
    }

    /// The panel's main text content: the end-of-session confirmation if
    /// present, else the live wrapping transcript — rendered inside the
    /// *same* bottom-anchored viewport (`render_transcript_viewport`) rather
    /// than a separately-laid-out slot. User correction after review: "✓
    /// Inserted" must replace the transcript organically, appearing exactly
    /// where its last line was, not jump to a different position.
    fn render_transcript_area(&self) -> AnyElement {
        let line = match &self.outcome {
            Some(outcome) => render_outcome_line(outcome),
            None => render_transcript_paragraph(&self.confirmed_text, &self.unconfirmed_text),
        };
        render_transcript_viewport(line)
    }

    /// Whether the overlay currently has anything session-related to show —
    /// true while recording, or while a `SessionCompleted`/`Error` outcome
    /// is still on screen. Drives the Full presentation's Overlay tab
    /// (`panel.rs`): live session content when true, the idle status block
    /// (driven by `StatusModel`) otherwise. `cfg`-gated: the Full
    /// presentation — and therefore this method's only caller — does not
    /// exist under `--features demo` (presentation never leaves Hud there).
    #[cfg(not(feature = "demo"))]
    #[must_use]
    pub(crate) fn has_session_content(&self) -> bool {
        self.recording || self.outcome.is_some()
    }
}

/// The shared, bottom-anchored transcript viewport (Fix 3): fixed height,
/// `overflow_hidden`, `justify_end`, fade-masked at the top. Both the live
/// transcript paragraph and the end-of-session outcome line render as
/// `content` here — the *same* frame, not two differently-positioned slots —
/// so swapping between them never shifts where the text sits.
fn render_transcript_viewport(content: AnyElement) -> AnyElement {
    div()
        .relative()
        .h(TRANSCRIPT_HEIGHT)
        .w_full()
        .overflow_hidden()
        .flex()
        .flex_col()
        .justify_end()
        .child(content)
        .child(fade_strip())
        .into_any_element()
}

/// Multi-line wrapping transcript paragraph (Fix 3): one `StyledText` run —
/// confirmed text solid, unconfirmed tail dimmed via a highlight range. Just
/// the text itself; `render_transcript_viewport` supplies the bottom-anchored
/// frame around it. `confirmed_len` is always a char boundary of
/// `full_text` because `full_text` is exactly `confirmed` followed by
/// `unconfirmed`, so highlighting `confirmed_len..full_text.len()` never
/// splits a UTF-8 code point.
fn render_transcript_paragraph(confirmed: &SharedString, unconfirmed: &SharedString) -> AnyElement {
    let full_text = format!("{confirmed}{unconfirmed}");
    let confirmed_len = confirmed.len();
    let paragraph = StyledText::new(full_text.clone()).with_highlights([(
        confirmed_len..full_text.len(),
        HighlightStyle {
            color: Some(color_unconfirmed_text()),
            ..Default::default()
        },
    )]);

    div()
        .text_size(TRANSCRIPT_TEXT_SIZE)
        .line_height(TRANSCRIPT_LINE_HEIGHT)
        .text_color(color_confirmed_text())
        .whitespace_normal()
        .child(paragraph)
        .into_any_element()
}

/// The top fade-out strip: a `linear_gradient` from the panel's own
/// background color (opaque) to fully transparent, so scrolled-past lines
/// melt away instead of slicing at the transcript viewport's edge.
fn fade_strip() -> impl IntoElement {
    div()
        .absolute()
        .top_0()
        .left_0()
        .w_full()
        .h(FADE_HEIGHT)
        .bg(linear_gradient(
            180.0,
            linear_color_stop(color_panel_bg(), 0.0),
            linear_color_stop(color_panel_bg().opacity(0.0), 1.0),
        ))
}

/// The end-of-session outcome line: outcome text, plus (for `Inserted` and
/// `ClipboardOnly` only — never `Failed`/`Error`) a "⌘V" key-cap chip with
/// outcome-specific wording (Fix 4). Sized to exactly one transcript line
/// (`TRANSCRIPT_LINE_HEIGHT`) so `render_transcript_viewport`'s
/// bottom-anchoring lands it in precisely the band the transcript's last
/// line occupied — no layout jump.
fn render_outcome_line(outcome: &Outcome) -> AnyElement {
    let (text, color) = outcome_text_and_color(outcome);
    let mut row = div()
        .flex()
        .items_center()
        .h(TRANSCRIPT_LINE_HEIGHT)
        .gap_2()
        .child(
            div()
                .text_size(TRANSCRIPT_TEXT_SIZE)
                .text_color(color)
                .child(text),
        );
    if let Some((hint, prominent)) = outcome.chip_hint() {
        row = row.child(key_cap_row(hint, prominent));
    }
    row.into_any_element()
}

/// The outcome's headline text + color — split out of `render_outcome_line`
/// to keep it under the 40-line render-helper limit (CONSTITUTION rule 28).
///
/// `Inserted`/`ClipboardOnly` share `color_confirmed_text()` — the exact
/// color of confirmed transcript text, not just the same hue (user
/// correction after review: separate elements by opacity/contrast and
/// typography first, hue last; the ✓ glyph already signals success, so the
/// headline no longer needs a saturated green to say it again). `Message`
/// (session errors, or the `Failed` injection outcome) keeps the slightly
/// dimmer `color_message()` — it's secondary information, not a headline.
fn outcome_text_and_color(outcome: &Outcome) -> (SharedString, Hsla) {
    match outcome {
        Outcome::Inserted => ("✓ Inserted".into(), color_confirmed_text()),
        Outcome::ClipboardOnly => ("Copied to clipboard".into(), color_confirmed_text()),
        // Secondary/in-progress information, like Message — not a headline.
        Outcome::Message { text, .. } => (text.clone(), color_message()),
    }
}

/// The "⌘V" key-cap chip + its instructional hint, dimmed or prominent
/// depending on the outcome (`Outcome::chip_hint`).
fn key_cap_row(hint: &'static str, prominent: bool) -> impl IntoElement {
    let opacity = if prominent {
        CHIP_OPACITY_PROMINENT
    } else {
        CHIP_OPACITY_DIMMED
    };
    div()
        .flex()
        .items_center()
        .gap_1p5()
        .opacity(opacity)
        .child(key_cap_chip())
        .child(
            div()
                .text_size(KEYCAP_TEXT_SIZE)
                .text_color(color_message())
                .child(hint),
        )
}

/// The bordered "⌘V" chip itself.
fn key_cap_chip() -> impl IntoElement {
    div()
        .px_1p5()
        .py_0p5()
        .rounded(px(theme::RADIUS_CHIP))
        .border_1()
        .border_color(hsla(
            NEUTRAL_HUE,
            NEUTRAL_SATURATION,
            LIGHTNESS_WHITE,
            KEYCAP_BORDER_OPACITY,
        ))
        .text_size(KEYCAP_TEXT_SIZE)
        .text_color(color_message())
        .child("⌘V")
}

/// The cosmetic activity waveform: `amps.len()` bars, centered, height
/// proportional to each bar's current amplitude. Ambient texture, not
/// information (user correction after review) — smaller bars, fainter
/// colors (`color_waveform_bar`), and `WAVEFORM_TOP_MARGIN` of breathing
/// room above it so it doesn't crowd the transcript.
fn render_waveform(amps: &[f32], recording: bool) -> impl IntoElement {
    div()
        .flex()
        .items_end()
        .justify_center()
        .gap_1()
        .mt(WAVEFORM_TOP_MARGIN)
        .h(WAVEFORM_HEIGHT)
        .children(amps.iter().map(|&amp| {
            let bar_height = px(amp * WAVEFORM_MAX_BAR_HEIGHT).max(WAVEFORM_BAR_MIN_HEIGHT);
            div()
                .w(WAVEFORM_BAR_WIDTH)
                .h(bar_height)
                .rounded(WAVEFORM_BAR_RADIUS)
                .bg(color_waveform_bar(recording))
        }))
}

/// The recording LED (Fix 5): warm red, breathing while recording via
/// `pulsating_between`; neutral gray and static while idle. Absolute
/// top-left inset of the panel, with a soft glow (`shadow`) only while lit.
fn render_recording_led(recording: bool) -> AnyElement {
    let dot = div()
        .absolute()
        .top(LED_INSET)
        .left(LED_INSET)
        .w(LED_SIZE)
        .h(LED_SIZE)
        .rounded_full()
        .bg(color_recording_dot(recording));

    if !recording {
        return dot.into_any_element();
    }

    dot.shadow(vec![BoxShadow {
        color: color_recording_glow(),
        offset: point(px(0.0), px(0.0)),
        blur_radius: LED_GLOW_BLUR,
        spread_radius: LED_GLOW_SPREAD,
    }])
    .with_animation(
        "recording-led",
        Animation::new(LED_PULSE_DURATION)
            .repeat()
            .with_easing(gpui::pulsating_between(LED_PULSE_MIN, LED_PULSE_MAX)),
        gpui::Styled::opacity,
    )
    .into_any_element()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare `OverlayModel` for tests that only exercise `handle_event`'s
    /// pure state transitions — constructed as a plain struct literal
    /// (private fields, same module) rather than through `OverlayModel::new`,
    /// which requires a live GPUI `Window`/`Context` this test module has no
    /// access to (and doesn't need: `new`'s only GPUI-dependent step is
    /// starting the animation loop, irrelevant here).
    fn test_overlay_model() -> OverlayModel {
        OverlayModel {
            recording: false,
            confirmed_text: SharedString::default(),
            unconfirmed_text: SharedString::default(),
            amplitudes: vec![0.0; BAR_COUNT],
            phases: [0.0; BAR_COUNT],
            activity_tick: 0.0,
            last_nudge: Instant::now(),
            outcome: None,
            outcome_since: Instant::now(),
            created_at: Instant::now(),
        }
    }

    // ── PartialTranscript rendering: producer-supplied fields, no re-derivation ──
    //
    // These replace the retired transcript-splitting unit tests (which
    // exercised a suffix-subtraction reconstruction that no longer exists —
    // `handle_event` now assigns `confirmed_text`/`unconfirmed_text`
    // straight from the event's own fields, so there is nothing left to
    // unit-test in isolation from `handle_event` itself; these assert the
    // field values land unmodified after a `PartialTranscript` event,
    // including the pathological case an index-math reconstruction would
    // have gotten wrong (`unconfirmed_text` not a suffix of anything, since
    // there is no combined field anymore for it to be a suffix of).

    #[test]
    fn partial_transcript_sets_confirmed_and_unconfirmed_verbatim() {
        let mut model = test_overlay_model();
        model.handle_event(DictationEvent::PartialTranscript {
            confirmed_text: "And so my dear".to_string(),
            unconfirmed_text: " friends".to_string(),
        });
        assert_eq!(model.confirmed_text.as_ref(), "And so my dear");
        assert_eq!(model.unconfirmed_text.as_ref(), " friends");
    }

    #[test]
    fn partial_transcript_tolerates_unconfirmed_text_unrelated_to_confirmed() {
        // With the retired combined-text shape, this exact case
        // (unconfirmed_text not a suffix of anything) was the "graceful
        // degradation" branch of the old suffix-subtraction split. With
        // producer-supplied fields there is no derivation to degrade — both
        // fields are simply stored as given, by construction.
        let mut model = test_overlay_model();
        model.handle_event(DictationEvent::PartialTranscript {
            confirmed_text: "hello world".to_string(),
            unconfirmed_text: "xyz".to_string(),
        });
        assert_eq!(model.confirmed_text.as_ref(), "hello world");
        assert_eq!(model.unconfirmed_text.as_ref(), "xyz");
    }

    // ── has_session_content — drives panel.rs's Full-presentation Overlay tab ──

    #[cfg(not(feature = "demo"))]
    #[test]
    fn has_session_content_false_when_idle() {
        let model = test_overlay_model();
        assert!(!model.has_session_content());
    }

    #[cfg(not(feature = "demo"))]
    #[test]
    fn has_session_content_true_while_recording() {
        let mut model = test_overlay_model();
        model.handle_event(DictationEvent::SessionStarted);
        assert!(model.has_session_content());
    }

    #[cfg(not(feature = "demo"))]
    #[test]
    fn has_session_content_true_while_an_outcome_is_on_screen() {
        let mut model = test_overlay_model();
        model.handle_event(DictationEvent::SessionCompleted {
            result: vuho_domain::TranscriptionResult {
                segments: vec![],
                full_text: "hello".to_string(),
                language: "en".to_string(),
            },
            injection: InjectionOutcome::Inserted,
        });
        assert!(!model.recording);
        assert!(model.has_session_content());
    }

    // ── Frozen-waveform regression (WP10, rule 22) ──────────────────────────

    /// Falsification target for the frozen-waveform bug: two calls at
    /// different elapsed times must yield different phases. Against the old
    /// buggy code (`now = Instant::now()` then immediately
    /// `now.elapsed()` in the same call, always ≈0), this assertion would
    /// fail — every call would see `elapsed_ms ≈ 0` regardless of how much
    /// real time had actually passed, and `wave_base` would return the same
    /// value both times (for a `bar_phase` where the sine term isn't at a
    /// coincidental turning point — `0.0` here isn't one: `sin` is strictly
    /// increasing near 0).
    #[test]
    #[allow(clippy::float_cmp)] // exact-inequality is the point: values must genuinely differ
    fn wave_base_changes_with_elapsed_time() {
        let bar_phase = 0.0;
        let at_0ms = wave_base(0.0, bar_phase);
        let at_500ms = wave_base(500.0, bar_phase);
        assert_ne!(
            at_0ms, at_500ms,
            "the waveform must actually animate over elapsed time, not stay frozen"
        );
    }

    #[test]
    fn wave_base_stays_in_unit_range() {
        for elapsed_ms in [0.0, 123.0, 10_000.0, 1_000_000.0] {
            for bar_phase in [0.0, 0.5, 1.0, 5.5] {
                let v = wave_base(elapsed_ms, bar_phase);
                assert!((0.0..=1.0).contains(&v), "wave_base out of range: {v}");
            }
        }
    }

    // ── Waveform decay tests ────────────────────────────────────────────────

    #[test]
    fn decay_no_elapsed_fully_active() {
        // With zero elapsed time, liveliness should be fully active (1.0).
        let liveliness = OverlayModel::compute_liveliness(1.0, 0.0);
        // baseline (0.05) + 1.0 * 0.95^0 = 0.05 + 1.0 = 1.05 → clamped to 1.0
        assert!((liveliness - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decay_exponential_drop() {
        // After 1 second with no new activity, liveliness should decay.
        // 0.95^(1.0 * 10.0) = 0.95^10 ≈ 0.5987
        // baseline (0.05) + 1.0 * 0.5987 ≈ 0.6487
        let liveliness = OverlayModel::compute_liveliness(1.0, 1.0);
        let expected = 0.05 + 1.0 * 0.95_f32.powf(10.0);
        assert!((liveliness - expected).abs() < 1e-6);
    }

    #[test]
    fn decay_long_idle_reaches_baseline() {
        // After 5 seconds, the decay term should be negligible.
        // 0.95^(5.0 * 10.0) = 0.95^50 ≈ 0.0769
        // baseline (0.05) + 0.0769 ≈ 0.1269
        let liveliness = OverlayModel::compute_liveliness(1.0, 5.0);
        let expected = 0.05 + 1.0 * 0.95_f32.powf(50.0);
        assert!((liveliness - expected).abs() < 1e-6);
    }

    #[test]
    fn decay_activity_tick_scales_linearly() {
        // Lower activity_tick should scale the decay proportionally.
        let liveliness_half = OverlayModel::compute_liveliness(0.5, 0.0);
        // At t=0: 0.5 * 1.0 + 0.05 = 0.55
        assert!((liveliness_half - 0.55).abs() < 1e-6);
    }

    #[test]
    fn decay_minimum_clamp() {
        // Even with zero activity_tick, the baseline should keep liveliness >= 0.02.
        let liveliness = OverlayModel::compute_liveliness(0.0, 0.0);
        assert!((liveliness - 0.05).abs() < 1e-6); // baseline, above the 0.02 clamp
    }

    #[test]
    fn decay_maximum_clamp() {
        // Very high activity_tick should be clamped to 1.0.
        let liveliness = OverlayModel::compute_liveliness(2.0, 0.0);
        // 0.05 + 2.0 * 1.0 = 2.05 → clamped to 1.0
        assert!((liveliness - 1.0).abs() < 1e-6);
    }

    // ── Per-outcome hide-delay tests (Fix 4) ────────────────────────────────

    #[test]
    fn outcome_hide_delay_inserted_is_short() {
        let delay = outcome_hide_delay(&InjectionOutcome::Inserted).unwrap();
        assert_eq!(delay, INSERTED_DISPLAY + HIDE_MARGIN);
        assert!(
            delay < CLIPBOARD_ONLY_DISPLAY + HIDE_MARGIN,
            "Inserted must be shorter than ClipboardOnly"
        );
    }

    #[test]
    fn outcome_hide_delay_clipboard_only_is_longer() {
        let delay = outcome_hide_delay(&InjectionOutcome::ClipboardOnly {
            reason: "secure input active".into(),
        })
        .unwrap();
        assert_eq!(delay, CLIPBOARD_ONLY_DISPLAY + HIDE_MARGIN);
    }

    #[test]
    fn outcome_hide_delay_failed_never_expires() {
        let delay = outcome_hide_delay(&InjectionOutcome::Failed {
            reason: "clipboard write failed".into(),
        });
        assert_eq!(delay, None);
    }

    #[test]
    fn outcome_for_injection_clipboard_only_maps_to_chip_hint() {
        let outcome = outcome_for_injection(&InjectionOutcome::ClipboardOnly {
            reason: "secure input active".into(),
        });
        let (hint, prominent) = outcome
            .chip_hint()
            .expect("ClipboardOnly must show a chip hint");
        assert_eq!(hint, "to paste");
        assert!(prominent);
    }

    #[test]
    fn outcome_for_injection_inserted_maps_to_dimmed_chip_hint() {
        let outcome = outcome_for_injection(&InjectionOutcome::Inserted);
        let (hint, prominent) = outcome.chip_hint().expect("Inserted must show a chip hint");
        assert_eq!(hint, "to paste again");
        assert!(!prominent);
    }

    #[test]
    fn outcome_for_injection_failed_never_shows_chip_hint() {
        // No false clipboard claim next to a failed clipboard write.
        let outcome = outcome_for_injection(&InjectionOutcome::Failed {
            reason: "clipboard write failed".into(),
        });
        assert!(outcome.chip_hint().is_none());
    }

    #[test]
    fn outcome_hide_delay_nothing_to_inject_auto_hides_briefly() {
        // A blank-transcript session: nothing was delivered and nothing is
        // at risk, so the note auto-clears after the default brief duration.
        let delay = outcome_hide_delay(&InjectionOutcome::NothingToInject).unwrap();
        assert_eq!(delay, DEFAULT_MESSAGE_DISPLAY + HIDE_MARGIN);
    }

    #[test]
    fn outcome_for_injection_nothing_to_inject_is_a_brief_note_without_chip() {
        // The clipboard was deliberately left untouched — showing a ⌘V chip
        // (or a persistent alarm) would be a lie about what happened.
        let outcome = outcome_for_injection(&InjectionOutcome::NothingToInject);
        assert!(matches!(
            &outcome,
            Outcome::Message {
                text,
                persistent: false
            } if text.as_ref() == "Nothing to insert"
        ));
        assert!(outcome.chip_hint().is_none());
    }
}
