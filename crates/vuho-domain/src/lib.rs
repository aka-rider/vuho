//! Pure domain types and events for Vuho speech-to-text.
//!
//! No external dependencies. Everything here is plain Rust — no FFI,
//! no platform-specific code, no framework coupling.

// ── Transcript data ──────────────────────────────────────────────────────────

/// A single segment of transcribed text with timing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptSegment {
    /// Segment index (monotonically increasing).
    pub id: u32,
    /// Text content of this segment.
    pub text: String,
    /// Start time in milliseconds (relative to session start).
    pub start_ms: u64,
    /// End time in milliseconds (relative to session start).
    pub end_ms: u64,
}

impl TranscriptSegment {
    /// Creates a segment from its parts.
    pub fn new(id: u32, text: impl Into<String>, start_ms: u64, end_ms: u64) -> Self {
        Self {
            id,
            text: text.into(),
            start_ms,
            end_ms,
        }
    }
}

/// Complete transcription result from the STT engine.
///
/// This shape is reused at two distinct points in the pipeline, each with a
/// different invariant on `full_text` — the type itself does not (and
/// cannot) distinguish which one a given value satisfies, so callers must
/// track it by context:
///
/// - As returned directly by `TranscriptionEngine::transcribe`/
///   `stop_stream`: `full_text` is exactly `segments` joined — the engine's
///   own raw assembly, not yet post-processed.
/// - As carried by `DictationEvent::SessionCompleted` once
///   `vuho-dictation`'s `DictationPipeline::emit_result` has finalized a
///   session: `full_text` is replaced by `vuho-postprocess`'s cleaned
///   output (filler words removed, spacing/newlines normalized), while
///   `segments` is left exactly as the engine produced it — raw,
///   unprocessed per-segment timing. Consumers that want the text that was
///   actually shown/injected must read `full_text`; consumers that want
///   segment-level timing unaffected by post-processing read `segments`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptionResult {
    /// Segments exactly as produced by the STT engine — never
    /// post-processed (see the struct-level doc comment).
    pub segments: Vec<TranscriptSegment>,
    /// The transcript text — the engine's raw concatenation of `segments`,
    /// or `vuho-postprocess`'s cleaned output once a session has been
    /// finalized (see the struct-level doc comment for which applies).
    pub full_text: String,
    /// Detected language code (e.g. `"en"`, `"vi"`, `"de"`), or `"und"`
    /// (ISO 639-2 "undetermined") if the producer could not determine one
    /// — never fabricated as a guess (CONSTITUTION rule 2). See
    /// `vuho-stt-engine`'s language-resolution doc comment for where the
    /// sentinel is produced, and `vuho-postprocess::postprocess`'s for how
    /// it's consumed (language-specific filler removal is skipped, generic
    /// formatting normalization still applies).
    pub language: String,
}

// ── Injection / error classification ────────────────────────────────────────

/// What happened to the finalized text after a session ended (ADR-012).
///
/// Deliberately NOT `#[non_exhaustive]` (ADR-018 fix — see that ADR's
/// consequence section): every consumer of this enum lives inside this
/// workspace, so a `match` without a wildcard arm is exactly what we want —
/// adding a variant here becomes a compile error at every downstream
/// `match` until it's handled deliberately, not a silent no-op behind a
/// `_` arm. `#[non_exhaustive]` would produce the opposite of that: it
/// exists for crates published to *other* crates' authors, forcing *them*
/// to add a wildcard arm (since they can't exhaustively enumerate variants
/// they don't control yet) — which is precisely the silent-swallow failure
/// mode this type must not have.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InjectionOutcome {
    /// Text was pasted into the focused app via synthesized ⌘V.
    Inserted,
    /// The paste keystroke could not be delivered (Secure Input active, or
    /// `CGEvent` synthesis failed) but the text is on the clipboard — the user
    /// must paste manually.
    ClipboardOnly {
        /// Human-readable explanation of why the keystroke could not be sent.
        reason: String,
    },
    /// Injection failed entirely: the clipboard write itself failed, so no
    /// copy of the text survives anywhere reachable by the user.
    Failed {
        /// Human-readable explanation of the failure.
        reason: String,
    },
    /// The session produced no text at all (a blank transcript — see
    /// [`is_blank_transcript`]), so injection was deliberately skipped: the
    /// clipboard and the focused app were left untouched.
    NothingToInject,
}

/// Category of session-level error (ADR-012), distinguishing cases the UI
/// must react to differently.
///
/// Not `#[non_exhaustive]` — see [`InjectionOutcome`]'s doc comment for why:
/// this is a workspace-internal type, and exhaustive matching is the point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// Microphone access was denied (TCC). Overlay should prompt to grant.
    MicPermissionDenied,
    /// Any other engine/OS failure not distinguished above.
    Other,
}

// ── Domain events ────────────────────────────────────────────────────────────

/// Events emitted by the dictation pipeline, consumed by the UI.
///
/// Not `#[non_exhaustive]`: the UI's `match` on this must handle a new
/// variant with a compile error, not a silent no-op wildcard arm — see
/// [`InjectionOutcome`]'s doc comment for the full rationale (this is a
/// workspace-internal type; `#[non_exhaustive]` would defeat, not enforce,
/// that goal).
#[derive(Clone, Debug)]
pub enum DictationEvent {
    /// Audio capture session started.
    SessionStarted,

    /// Partial transcript update — streaming mode.
    ///
    /// Both fields are producer-supplied (the STT engine's `Accumulator`
    /// owns the confirmed/unconfirmed split — see
    /// `vuho-stt-engine/src/stream/accumulator.rs`): the UI renders them
    /// directly and never re-derives one from the other by subtracting a
    /// suffix, which was a rule-2 violation this event shape used to force
    /// (a predecessor field bundled confirmed and unconfirmed text
    /// together, so a consumer that only wanted the confirmed prefix had to
    /// peel `unconfirmed_text` back out of it itself).
    PartialTranscript {
        /// The confirmed (committed) transcript text so far.
        confirmed_text: String,
        /// The unconfirmed (not-yet-committed) tail, if any.
        unconfirmed_text: String,
    },

    /// Cosmetic activity level for waveform visualization.
    ///
    /// Derived from capture-thread RMS (`vuho-audio`), resampled to 16 kHz
    /// mono — not a per-model-frame energy value. A visual proxy for the
    /// waveform animation (ADR-001).
    ///
    /// The value is in `[0.0, 1.0]`, a single float (not a vector) because
    /// streaming mode only produces one activity sample per callback invocation.
    Activity {
        /// Normalized activity level in `[0.0, 1.0]`.
        level: f32,
    },

    /// Final transcription completed.
    SessionCompleted {
        /// The finalized (cleaned) transcription.
        result: TranscriptionResult,
        /// What happened to the text afterward (pasted, clipboard-only, or failed).
        injection: InjectionOutcome,
    },

    /// An error occurred during the session.
    Error {
        /// Human-readable error description.
        message: String,
        /// Whether the session can be retried without full restart.
        recoverable: bool,
        /// Category of failure, distinguishing UI reactions (ADR-012).
        kind: ErrorKind,
    },
}

// ── Transcript cleanup ───────────────────────────────────────────────────────

/// Whether dictated text is blank — empty or whitespace-only.
///
/// Used by `vuho-dictation`'s injection gate: blank text must not be
/// injected into the focused app (which would clobber the user's clipboard
/// with an empty string).
#[must_use]
pub fn is_blank_transcript(text: &str) -> bool {
    text.trim().is_empty()
}

// ── Model provisioning ───────────────────────────────────────────────────────

/// Readiness of the on-disk STT model, as produced by `vuho-model-fetch`'s
/// `availability()` chokepoint and consumed by `vuho-ui` to drive the
/// provisioning UI (download button, progress text, engine warmup gate).
///
/// Deliberately NOT `#[non_exhaustive]` — see [`InjectionOutcome`]'s doc
/// comment for the full rationale (this is a workspace-internal type: every
/// consumer lives inside this workspace, so a new variant must be a compile
/// error at every `match`, not a silent `_` arm — ADR-018).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelStatus {
    /// No usable model is present on disk (the model directory does not
    /// exist, or exists but is empty/incomplete — see `vuho-model-fetch`'s
    /// `availability()` for the exact detection rule). `total_bytes` comes
    /// from the repo-pinned `models.lock.json`, not from any network call,
    /// so the UI can show the download size (e.g. "474 MB") before a
    /// download has ever started.
    ///
    /// An unreadable or permission-denied model directory is [`Self::Failed`],
    /// not `Missing` — conflating the two would present a broken
    /// `~/Library/Application Support` as "click Download" and then fail
    /// identically in a loop instead of surfacing the real cause
    /// (CONSTITUTION rule 2: don't fabricate a fact the producer doesn't
    /// actually have).
    Missing {
        /// Total download size in bytes, from the pinned lockfile.
        total_bytes: u64,
    },
    /// A download is in progress.
    ///
    /// `received_bytes` is expected to be monotonically non-decreasing and
    /// `<= total_bytes` for the lifetime of a single download — the
    /// producer (`ChannelProgress` in `vuho-model-fetch`) is written to
    /// maintain that, but this type does not itself enforce it (no
    /// validation on construction); a consumer must not panic if it sees a
    /// value that briefly violates it.
    Downloading {
        /// Bytes received so far, summed across all files in the download.
        received_bytes: u64,
        /// Total bytes to receive, from the pinned lockfile.
        total_bytes: u64,
    },
    /// The download finished and the model is being verified against the
    /// pinned lockfile (hashes for a fresh download; sizes/revision for a
    /// quick startup check — see `vuho-model-fetch`'s `availability()`).
    Verifying,
    /// The model is present, verified, and safe to load.
    Ready,
    /// Provisioning failed: I/O error, network error, or a hash/size
    /// mismatch against the pinned lockfile. This is for real errors only —
    /// a model that is simply absent is [`Self::Missing`], not `Failed`; see
    /// that variant's doc comment for why the distinction matters.
    Failed {
        /// Human-readable explanation of the failure.
        message: String,
    },
}

impl ModelStatus {
    /// Download progress as a fraction in `[0.0, 1.0]`, for progress-bar
    /// rendering.
    ///
    /// Returns `None` for every variant except [`Self::Downloading`], and
    /// also `None` when `total_bytes == 0` (nothing to divide by, and a
    /// zero-byte model is not a real download to show progress for).
    #[must_use]
    pub fn fraction(&self) -> Option<f32> {
        match *self {
            Self::Downloading {
                received_bytes,
                total_bytes,
            } if total_bytes > 0 => {
                #[allow(clippy::cast_precision_loss)]
                let fraction = received_bytes as f32 / total_bytes as f32;
                Some(fraction)
            }
            _ => None,
        }
    }
}

// ── Commands ─────────────────────────────────────────────────────────────────

/// Commands sent to the dictation pipeline from the UI / hotkey handler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DictationCommand {
    /// Start a new dictation session.
    Start,
    /// Stop the current session and finalize transcription.
    Stop,
    /// Toggle the session: start if idle, stop if recording.
    ///
    /// Used by the global hotkey (ADR-007: `CapsLock` tap-to-toggle).
    /// The pipeline derives the action from the current state.
    Toggle,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_blank_transcript_true_for_empty_string() {
        assert!(is_blank_transcript(""));
    }

    #[test]
    fn is_blank_transcript_true_for_whitespace_only() {
        assert!(is_blank_transcript(" \t\n  "));
    }

    #[test]
    fn is_blank_transcript_false_for_normal_text() {
        assert!(!is_blank_transcript("hello world"));
    }

    #[test]
    fn fraction_computes_ratio_while_downloading() {
        let status = ModelStatus::Downloading {
            received_bytes: 25,
            total_bytes: 100,
        };
        assert_eq!(status.fraction(), Some(0.25));
    }

    #[test]
    fn fraction_none_when_total_bytes_zero() {
        let status = ModelStatus::Downloading {
            received_bytes: 0,
            total_bytes: 0,
        };
        assert_eq!(status.fraction(), None);
    }

    #[test]
    fn fraction_none_for_non_downloading_variants() {
        assert_eq!(ModelStatus::Missing { total_bytes: 100 }.fraction(), None);
        assert_eq!(ModelStatus::Verifying.fraction(), None);
        assert_eq!(ModelStatus::Ready.fraction(), None);
        assert_eq!(
            ModelStatus::Failed {
                message: "oops".into()
            }
            .fraction(),
            None
        );
    }
}
