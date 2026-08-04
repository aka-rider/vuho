# Vuho Architecture — Decisions & Glossary

> Canonical architecture record for Vuho, produced via a `/grill-with-docs` session
> (one-decision-at-a-time interview → ADRs + glossary). **This document supersedes the
> aspirational parts of `scratchpad.md` wherever they conflict with a decision recorded here.**

## Context

Vuho is a local-first, fully-private WisprFlow-style dictation app for Apple Silicon macOS
(ANE + Metal). **This document's early ADRs (001, 003, 005, 010) describe the original
architecture: Rust `TranscriptionEngine` → `libloading` → a Swift dylib → WhisperKit → CoreML.**
That architecture was built, then torn out and replaced with a pure-Rust stack — `cpal`
audio capture, a hand-rolled Parakeet-TDT decoder over native CoreML (`objc2-core-ml`), no
Swift, no FFI boundary at all. The superseded ADRs are kept verbatim below (with a banner)
for historical record; ADR-013/014/015 record what was actually built and learned instead.
Streaming (live partial transcripts) is mid-flight in a parallel workstream — see ADR-015 for
what's shipped vs. still in progress. The ADRs below resolve the open architectural forks.

---

## Glossary (ubiquitous language)

- **WhisperKit engine** *(retired terminology — see ADR-014)* — the Swift dynamic library that
  used to expose a C ABI (`@_cdecl`) loaded by Rust at runtime via `libloading`. Fully removed;
  kept here only because older ADR text below still uses the term.
- **STT engine port** — the Rust `TranscriptionEngine` trait that dictation orchestration talks
  to. `ParakeetEngine` (native CoreML, in-process — no dylib, no FFI) is now the only
  implementation; the trait remains because ADR-004's variation-is-real argument still holds
  (nothing else changed), not because a second engine is imminent.
- **Session** — one dictation episode: hotkey-start → speak → hotkey-confirm-end → cleanup → inject.
  Boundaries are **explicit (hotkey-driven)**, not VAD-driven.
- **Committed / fresh tokens** — the current streaming/windowing vocabulary (replaces the old
  WhisperKit-specific "confirmed/unconfirmed segments" idea at the decoder level): `committed`
  is the token sequence already folded into the transcript by `stream::merge`; `fresh` is a
  window's newly-decoded tokens before overlap dedup. At the UI/event level,
  `DictationEvent::PartialTranscript { current_text, unconfirmed_text }` still carries a
  stable/live-tail distinction — see ADR-015.
- **Activity waveform** — in the target architecture this is **real** audio RMS from the
  `vuho-audio` capture thread (`CaptureHandle::level_rms`), not a cosmetic animation (ADR-013
  supersedes ADR-001 on this point). The GPUI overlay layer still applies decay/jitter for
  visual smoothing on top of that real signal.
- **Partial transcript** — the live `currentText` (confirmed + unconfirmed) surfaced to the overlay.
- **Overlay** — the always-on-top, semi-transparent, non-focus-stealing, click-through panel showing
  the partial transcript + activity waveform. Must NOT take focus (so injection lands in the prior app).

---

## Decisions (ADRs)

### ADR-001 — WhisperKit owns the microphone; waveform is a cosmetic animation

**Status:** Superseded by ADR-013 (audio capture) and ADR-014 (STT engine). WhisperKit was
retired in its entirety; Rust (`vuho-audio`) now owns the microphone and the waveform is
driven by real capture-thread RMS, not a cosmetic animation. Kept verbatim below for
historical record.

**Decision:** The engine wraps WhisperKit's `AudioStreamTranscriber` using its default
`AudioProcessor` (AVAudioEngine captures the mic internally). Streaming, VAD-based chunking,
and segment confirmation are all delegated to WhisperKit. The overlay waveform is a cosmetic
activity animation reacting to streaming-update events, not real per-frame energy.

**Consequences:**
- The Rust `vuho-audio` cpal capture + rubato resampler are **vestigial for STT** (retired in ADR-010).
- `vad-rs` is **not needed** — WhisperKit's internal VAD handles chunking.
- Mic-permission handling moves to the Swift side.
- The engine needs a **streaming API** (start/stop + state callback), replacing/augmenting the
  current batch-only `vuho_transcribe`.
- Domain event `WaveformUpdate { samples }` will be reshaped toward an activity signal, not raw samples.
- **Device selection** (settings-window microphone dropdown) stays Swift-side too: `vuho_start_stream`
  takes a nullable device-name param, resolved to a `DeviceID` via `AudioProcessor.getAudioDevices()`
  (unstable IDs across reboots — the name is what's persisted). Because `AudioProcessor`'s
  `startRecordingLive`/`startStreamingRecordingLive`/`resumeRecordingLive` live in a `public extension`
  on the concrete class (not overridable by subclassing), selection needs a small wrapper —
  `StreamAudioProcessor: AudioProcessing` — that forwards every member and substitutes the
  configured `DeviceID` on the three device-taking calls. `vuho_list_input_devices` (static, no
  `vuho_init` needed) enumerates device names for the dropdown.
- **A stream handle means the microphone is live.** Because Swift owns the mic, Rust cannot check
  for itself whether audio is really flowing — it can only believe what the FFI tells it. So the
  boundary must not be able to lie: `vuho_start_stream` returns a non-zero handle **only** if the
  audio engine was confirmed running, and reports every failure via `out_status`
  (`vuho_stream_status_t`) + `out_error`. This needs a `StartGate`, because
  `AudioStreamTranscriber.startStreamTranscription()` conflates *start* and *run* (it doesn't
  return until the stream ends, and its `state.isRecording` is set *before* the throwing call and
  never reset — so neither awaiting it nor polling it can report a start). `StreamAudioProcessor`
  is therefore wrapped **unconditionally**, even with no device pinned: it sits on the real
  `startRecordingLive` call, which is the only honest place to observe the outcome.
- **Post-start deaths get their own channel.** A stream that dies mid-session cannot report through
  the state callback — the absence of state callbacks *is* the failure. Hence
  `vuho_stream_error_cb`, which Rust turns into `DictationEvent::Error`, aborting the session back
  to `Idle`. Note WhisperKit's `realtimeLoop` catches its own errors and `break`s, so a mid-stream
  death returns *normally*: detecting it needs a stop-flag comparison, not a `catch`.
- **No silent-stream watchdog.** A "live but quiet for N seconds" timer was rejected: it re-detects,
  more weakly, what the gate and error callback already report at the source with the real cause,
  and it would be calibrated against WhisperKit's internal loop timing. A user thinking for 20s is
  a valid session, not an error.

**Rejected alternatives:** (a) Rust owns mic and pushes audio to a custom `AudioProcessing`
conformer — more control but a high-rate audio-push FFI path and a custom AudioProcessor to
maintain; (b) Rust-driven batch chunking reinventing confirmed/unconfirmed tracking — a real
quality risk, reimplementing the one hard thing WhisperKit already solves.

### ADR-002 — macOS OS-integration lives in Rust via `objc2`, not Swift, not `arboard`

**Status:** Accepted.

**Decision:** Global hotkey, text injection (NSPasteboard + CGEvent Cmd→V), and TIS
keyboard-language detection are implemented in Rust using the `objc2` family
(`objc2-app-kit`, `objc2-foundation`, `objc2-core-graphics`, plus Carbon/TIS as needed).
`arboard` is removed (resolves CONSTITUTION rule 7). **objc2-first**: only introduce an
additional, separate Swift library for a specific concern (e.g. injection or hotkey) if
objc2 genuinely cannot do it — not preemptively.

**Consequences:**
- *(Historical, at time of writing:)* the only Swift in the project was the WhisperKit engine
  (ADR-003); OS integration added no Swift. **Since superseded:** the WhisperKit engine was
  itself retired (ADR-014), so the project now has zero Swift/FFI anywhere — `objc2` is the
  only way any macOS API is called, in every crate, with no exception.
- `vuho-os-integration` gains real objc2 impls; `arboard` dep dropped; `objc2-*` become direct deps.
- Global hotkey works app-unfocused → likely CGEventTap (needs Accessibility permission) or a
  Carbon `RegisterEventHotKey`; to be settled during implementation. (See ADR-007 for the CapsLock wrinkle.)

### ADR-003 — Retire "shim"; it is the WhisperKit adapter behind an STT engine port

**Status:** Superseded by ADR-014. The WhisperKit adapter this ADR names no longer exists —
the engine is native CoreML in-process, with no Swift package/dylib/`@_cdecl` boundary to
rename at all. The renaming decision (retire "shim") remains historically correct; the thing
it renamed is gone. Kept verbatim below for historical record.

**Decision:** Rename away from "shim"/`VuhoShim`. The Swift dylib is the **WhisperKit STT
engine adapter**; Rust orchestration depends on an **STT engine port** (trait), with WhisperKit
as the first implementation. This keeps the door open for alternative engines without touching
the pipeline.

**Consequences:**
- Swift package/product/dylib and the `@_cdecl` symbol prefix get renamed (e.g. `WhisperKitEngine`
  / `vuho_whisperkit_*`); Rust FFI wrapper renamed to match.
- Introduces a real `TranscriptionEngine` trait; the FFI wrapper becomes one adapter. Port scope: ADR-004.

### ADR-004 — Abstraction only where variation is real: STT port, concrete everywhere else

**Status:** Accepted.

**Decision:** Introduce exactly one port — the `TranscriptionEngine` trait — because STT engines
are genuinely plural. Audio, text injection, global hotkey, and language detection stay concrete
macOS structs (single platform, single implementation). No full hexagonal port layer. YAGNI:
abstraction is earned by real variation, not symmetry.

**Consequences:**
- `vuho-stt-engine` exposes a `TranscriptionEngine` trait; the WhisperKit FFI wrapper implements it.
- Pipeline orchestration depends on the trait, not the concrete WhisperKit type.
- No `TextInjector`/`HotkeyListener`/`LanguageDetector` traits — direct concrete calls.

### ADR-005 — Build & FFI mechanics (delegated judgment)

**Status:** Superseded by ADR-014. There is no `swift build` step, no dylib, no `libloading`,
no C ABI boundary in the shipped architecture — `cargo build --release -p vuho-ui` requires
no Swift toolchain at all. Kept verbatim below for historical record.

**Decision:** `build.rs` automates `swift build` for the WhisperKit engine so `cargo build`
produces a current dylib (no more manual `cd shim && swift build`). Keep **runtime loading via
`libloading`** — it already works and cheaply supports the "swap the engine dylib" idea the STT
port opens, with no compile-time coupling to the Swift toolchain. Make dylib discovery
**app-bundle-aware** (search the bundle's `Frameworks/` dir; fall back to the dev build path).
**Drop `bindgen`** (unused; a ~6-function C ABI is cleaner hand-written and kept in sync with the
one hand-written header). Rationale: minimal moving parts, robust when shipped in a `.app`.

**Consequences:** `bindgen` removed from workspace deps; `build.rs` gains a swift-build step
(with a guard so non-macOS / CI-without-Swift degrades gracefully); dylib is bundled at package time.

### ADR-006 — Overlay: GPUI, decided by a PoC spike against explicit fork criteria

**Status:** Accepted (approach); overlay tech **pending a PoC**.

**Decision:** Attempt the overlay in GPUI (per CLAUDE.md) and **fork on a PoC**, not on faith.
Confirmed-supported primitives make GPUI viable: `WindowKind::PopUp`
(= macOS `NSWindowStyleMaskNonactivatingPanel`, non-activating), `focus: false`, and
`window_background: Transparent/Blurred`. The PoC must resolve what the public `WindowOptions`
API does NOT expose, using objc2 surgery on the underlying NSWindow if GPUI hands it out.

**PoC go/no-go criteria (in priority order):**
1. Show a transparent `PopUp` window that does **not** steal focus/activation from the frontmost app
   (verify by injecting text into another app while the overlay is visible).
2. Make it **click-through** (`setIgnoresMouseEvents(true)`) — requires reaching the NSWindow.
3. Raise it **above the menu bar / fullscreen spaces** (window level + `collectionBehavior`) via objc2.
4. Acceptable render/animation performance at overlay size.

**Fork:** criteria 1 met (2–3 achievable via objc2) → **GPUI**. Otherwise → **native NSPanel via
objc2** (fallback; total control, reuses the objc2 stack from ADR-002, no Metal-toolchain fight).
First resolve the current blocker: GPUI is commented out because the Metal/Xcode toolchain won't
build — the spike starts by getting GPUI to compile at all.

**Known risks:** no public field for level/click-through/collectionBehavior; popup-window freeze
bug [zed#42821](https://github.com/zed-industries/zed/issues/42821); GPUI may not expose its
NSWindow handle (would force the fallback). Transparency support: [zed#9610](https://github.com/zed-industries/zed/pull/9610).

### ADR-007 — Session control: tap CapsLock (default, configurable) toggles recording

**Status:** Accepted.

**Decision:** A single global hotkey — **CapsLock by default, configurable** — with a **tap-to-toggle**
gesture: tap to start recording (a "green light" indicator shows the recording state), tap again to
confirm end → cleanup → inject. No push-to-talk/hold mode. Two states (idle / recording), one gesture.

**Consequences:**
- `DictationCommand` gains `Toggle`, used by the chord preset and the status-bar menu — both are
  momentary gestures with no external state to stay in phase with.
- **CapsLock is level-triggered, not edge-triggered** (amended after a field bug: a dropped/discarded
  command on any of several paths — warmup discarding a command while the LED had already toggled, a
  status-bar Stop that doesn't touch the LED, a mic-permission failure that leaves the pipeline `Idle`
  with the LED lit — inverted the tap-to-toggle mapping for the rest of the run: CapsLock-off started
  dictation and CapsLock-on stopped it). The LED is the source of truth: rising edge (latch now set) →
  `DictationCommand::Start`, falling edge (now clear) → `DictationCommand::Stop`
  (`caps_lock_command` in `vuho-os-integration/src/hotkey.rs`). The pipeline's `dispatch` already
  no-ops `(Start, Recording)` and `(Stop, Idle)`, so any desync self-heals at the next tap instead of
  persisting — sending an undifferentiated `Toggle` (whose start-vs-stop meaning depends on which edge
  produced it, invisible to the pipeline) is exactly what made a single dropped command permanent.
  LED → app only: the app does not write the CapsLock LED via IOKit, so a status-bar Stop can leave the
  LED lit; the next tap-off is then a harmless no-op rather than a spurious start.
- The overlay's "green light" is the recording-state indicator (distinct from the activity waveform).
- **CapsLock capture constraint:** CapsLock arrives as a `flagsChanged` event, not a normal keydown;
  intercepting it AND suppressing the native caps-lock toggle typically needs an **IOKit HID tap** or a
  **hidutil remap**, not a plain CGEventTap. This is the most likely place objc2 alone may be
  insufficient — the first candidate for the ADR-002 "add a small Swift lib only if objc2 fails" escape.
  A configurable key lets us fall back to a conventional chord (e.g. ⌥-Space) if CapsLock proves too costly.
  (Suppressing the native LED toggle remains out of scope — see the LED → app only note above.)

### ADR-008 — Bundle the CoreML model in the .app (offline from first launch)

**Status:** Superseded by ADR-020. The "no download, ever" invariant no longer holds for the
Homebrew-cask distribution shape — the model becomes a first-run download into a user-data
directory, verified against a repo-pinned lock. The DMG build still embeds the model exactly as
described here; the invariant now applies only to that trust boundary, not universally. Kept
verbatim below for historical record.

**Decision:** Embed the Parakeet-TDT CoreML model
(`FluidInference/parakeet-tdt-0.6b-v3-coreml`, itself a CoreML export of
`nvidia/parakeet-tdt-0.6b-v3`, ~490 MB) in the app bundle. No download, no network, ever. The
engine loads it from the bundle path via `resolve_model_folder()` (unchanged chokepoint,
below). *(Historical: the original decision named the WhisperKit `large-v3-turbo` model,
~1.5 GB; that engine and model are retired — see ADR-014.)*

**Consequences:**
- ~500–600 MB `.app` (down from the WhisperKit-era ~1.5 GB estimate); model updates ship as
  app updates. Fixed model for v1 (no selector — YAGNI).
- The dev-time model dir (`models/`) must NOT be committed to git (too large, gitignored) —
  it's a **packaging input**, fetched by `./scripts/fetch-model.sh` (idempotent,
  `huggingface-cli` with a `curl` fallback, pinned upstream revisions) and copied into the
  bundle's `Contents/Resources/` by `scripts/bundle-macos.sh` at package time.
- **Removes** the first-run-download failure path entirely from error recovery.

**Enforcement:** "no download, ever" is a property of the code, not a convention — unchanged
by the engine swap. Resolution lives in one Rust chokepoint —
`vuho_stt_engine::resolve_model_folder()` (`VUHO_MODEL_FOLDER` → bundle `Contents/Resources/`
→ workspace dev dir (`models/<VUHO_MODEL_NAME>`) → `ModelFolderMissing`) — and
`ParakeetEngine::load` takes the resolved folder, so an engine without one cannot be
constructed. Nothing in the CoreML loading path (`coreml.rs`) reads environment or falls back
to a network fetch; a missing model is an error at the chokepoint, full stop. *(Historical:
the retired Swift/WhisperKit side used to resolve paths itself and silently attempt a network
fetch into `~/Documents/huggingface/` on a plain `cargo run` — the bug this ADR's original
enforcement note was written to explain. That failure mode is structurally impossible now:
there is no Swift code left to resolve anything.)*

### ADR-009 — Language: keyboard input method is authoritative (delegated judgment)

**Status:** Accepted.

**Decision:** Per spec ("STT language always matches OS-native keyboard input method"), query
`TISCopyCurrentKeyboardInputSource` → `kTISPropertyInputSourceLanguages` (objc2/Carbon) at session
start, map to an engine-supported language code, and pass it to `TranscriptionEngine::transcribe`
as `Option<&str>`. *(Historical: this ADR originally named `DecodingOptions.language` and a
WhisperKit-auto-detect fallback — both WhisperKit-specific and retired with it, ADR-014.)*
**Current state:** the keyboard-language-is-authoritative decision stands and the language plumbs
through as real data (CONSTITUTION rule 2), not a hardcoded `"en"`; `ParakeetEngine::transcribe`
itself has no built-in auto-detect equivalent to fall back to yet, so an unrecognized keyboard
language currently falls back to `"en"` at the call site rather than to a model-level detector —
a smaller gap than the original ADR anticipated, tracked as a later refinement, not a defect.

### ADR-010 — Engine exposes batch AND streaming; Rust audio path retired (delegated judgment)

**Status:** Superseded by ADR-013. The premise (WhisperKit owns the mic, so `vuho-audio` is
dead weight) no longer holds — WhisperKit is gone, and with no engine left to own the
microphone internally, `vuho-audio` was **reinstated** as the sole owner of the audio path.
`cpal`, `rubato`, and a VAD (`voice_activity_detector`, Silero-based, replacing the
never-implemented `vad-rs` this ADR names) are all back in the dependency tree, by design.
Kept verbatim below for historical record.

**Decision:** The WhisperKit engine adapter exposes two entrypoints behind the `TranscriptionEngine`
trait: (1) **streaming** (`start`/`stop` + a state callback) — the product path (ADR-001); (2) a
**batch/file** transcribe — the deterministic, mic-less regression test path (replacing today's
`test-stt-ffi` buffer path with a file path, so no Rust-side audio handling is needed for tests).

**Consequences:**
- `vuho-audio` (cpal capture + rubato resampler + `AudioCapture`) is **deleted** — WhisperKit owns
  the mic and does its own audio loading; `cpal`, `rubato`, `vad-rs` drop out of the dep tree.
- WAV/PCM parsing in Rust is no longer needed (batch test passes a file path to WhisperKit); the
  `parse_wav_data`/`pcm_i16_to_f32` helpers retire with the crate.
- The engine's blocking `syncOnMain` (semaphore) stays fine for batch; **streaming** uses a persistent
  callback that marshals `AudioStreamTranscriber` state → a crossbeam channel → `DictationEvent`s,
  never blocking an OS thread.

### ADR-011 — End-of-session flow: post-process → clipboard → Cmd→V into prior app (delegated judgment)

**Status:** Accepted; amended by ADR-017 (adopted LLM cleanup, later reverted) — the cleanup step is now rule-based again.

**Decision:** On the second CapsLock tap: take the STT engine's final transcribed text →
post-process (`vuho_postprocess::postprocess(text, language)`) → write to `NSPasteboard` (objc2) → synthesize
Cmd→V via `CGEvent` into the previously-focused app. The overlay never takes focus (ADR-006), so
the paste lands in the right target. Optional start/stop earcons via `rodio`. *(Historical: the
cleanup step was originally `vuho-postprocess`'s rule-based `postprocess(text,
language)`. ADR-017 replaced it wholesale with an on-device LLM (LFM2.5-1.2B via Candle); that
experiment was reverted — rule-based post-processing remains.)*

**Consequences:** `vuho-os-integration` gains `inject_text` (currently missing despite an
`InjectionFailed` error variant); `arboard` replaced by objc2 `NSPasteboard`.

### ADR-012 — Error recovery matrix (delegated judgment)

**Status:** Accepted; amended for the Rust-owned audio path (ADR-013) — mic-permission
detection moved from "only discoverable via a failed engine start" to a direct, synchronous
Rust-side check; amended again for the model-download path (ADR-020), which turns the
model-download row below from N/A into a real recovery case.

- **Mic permission denied** → detected **Rust-side**, not inferred from an engine failure:
  `vuho_audio::permission::mic_authorization_status()` calls `AVCaptureDevice::
  authorizationStatusForMediaType` (objc2-av-foundation) directly. `Denied`/`Restricted` →
  `EngineError::MicPermissionDenied` → `DictationEvent::Error { recoverable: false }`, overlay
  prompts to grant; `NotDetermined` triggers `requestAccessForMediaType_completionHandler`
  (fire-and-forget — the modal system dialog can block on user input arbitrarily long, so the
  pipeline doesn't await it, it just re-checks on the next session start).
- **Injection blocked by Secure Input** (password fields) → leave text on clipboard + toast "copied,
  paste manually"; `recoverable: true`.
- **Engine load / ANE failure** → surface error. *(Historical: this line originally anticipated
  a WhisperKit GPU↔CPU compute-unit fallback knob. The retired engine's fallback concept doesn't
  apply to the current one: `ParakeetModels::load` assigns compute units per component at load
  time — `CpuOnly` for `Preprocessor`/`ParakeetDecoder`/`RNNTJoint`, `CpuAndNeuralEngine` for
  `ParakeetEncoder_15s` — as a fixed property of each `.mlmodelc`'s shape flexibility (see
  ADR-014), not a runtime-selectable fallback.)*
- **CapsLock capture unavailable** (no Accessibility/HID permission) → prompt for permission; allow
  reconfiguring to a conventional chord (ADR-007).
- **Model-download failures** (ADR-020) → network failure, hash mismatch, or insufficient disk
  space during `vuho-model-fetch::download` all map to `ModelStatus::Failed { message }`,
  surfaced in the readiness window (ADR-020's generalization of this ADR-016 gate) with a Retry
  button that re-issues the same download. A partial download lives at `<dir>.partial` and is
  never promoted to `<dir>` — the rename only happens after `Full` verification against
  `models.lock.json` succeeds — so a crash mid-download leaves `availability()` reporting
  `Missing`, plus a leftover `.partial`. `download()` removes that leftover `.partial` outright
  before the next attempt's transfer, rather than handing it to `hf-hub` to "resume": `hf-hub`
  1.0's `local_dir` mode skips any destination file that merely exists, with no size/hash check,
  so a truncated leftover would otherwise be silently treated as already-downloaded and fail
  verification identically on every retry. Never a corrupt `Ready`.

### ADR-013 — Rust owns the audio path; `vuho-audio` reinstated

**Status:** Accepted. Supersedes the mic-ownership consequence of ADR-001 and the
audio-path deletion in ADR-010.

**Decision:** With no engine left that owns the microphone internally (WhisperKit is gone —
ADR-014), Rust owns audio capture end-to-end via a **reinstated** `vuho-audio` crate. (ADR-010
deleted a predecessor crate of the same name and purpose; this is not a resurrection of old
code but a fresh implementation earning its place back on its own merits, once the premise
that justified deleting it — "WhisperKit owns the mic" — no longer held.)

- A dedicated `"vuho-audio-capture"` thread owns the `!Send` `cpal::Stream` for its entire
  lifecycle — build, `play()`, pump, drop — all on that one thread (CONSTITUTION rule 1: one
  owner per resource). The realtime audio callback does the minimum possible: convert to
  `f32` if needed, push into an `rtrb::Producer` ring buffer, and compute a block RMS into an
  `AtomicU32` (via `f32::to_bits`) for the waveform — no allocation, no resampling, nothing
  that can block or jitter on the realtime thread.
- The pump loop (same thread) drains the `rtrb::Consumer`, downmixes N channels to mono,
  feeds `rubato::FftFixedIn` to resample to 16 kHz (passthrough if the device is already
  16 kHz), and forwards fixed chunks over a bounded `crossbeam_channel` to the engine.
- `CaptureHandle::stop()` sets an `AtomicBool` and joins the thread (CONSTITUTION rule 9: the
  stopper owns the stop signal) — the thread observes the flag, drops the `cpal::Stream` on
  itself, flushes the resampler tail, and closes the chunk channel as the end-of-audio signal.
- **The waveform is now real, not cosmetic** (reverses the ADR-001 consequence):
  `CaptureHandle::level_rms()` is genuine per-block RMS of captured audio, not a
  streaming-update-driven animation. The GPUI overlay layer still applies decay/jitter for
  visual smoothing (see `crates/vuho-ui/src/overlay.rs`), but the input signal underneath it
  is real audio energy.
- **Device selection** stays name-based (unchanged decision from ADR-001, moved implementation):
  `cpal::default_host().input_devices()` filtered by device description name; an unresolved
  configured name falls back to the system default input device with a `log::warn!`, mirroring
  the old Swift-side behavior's spirit without a `DeviceID`-vs-name instability concern (cpal
  resolves fresh from the name every session start, no cached ID to go stale).
- **Mic permission is now Rust-side**, not Swift-side: `vuho_audio::permission` queries
  `AVCaptureDevice::authorizationStatusForMediaType` directly via `objc2-av-foundation` (see
  ADR-012's amendment).

**The "live handle means audio flowing" invariant is preserved, in-process.** ADR-001 needed a
`StartGate` specifically because Rust could only trust what an external Swift process *claimed*
over an FFI boundary that could lie by omission. That specific problem — a boundary that can't
be introspected — no longer exists (there is no boundary), but the invariant itself is still
worth keeping honest: `vuho_audio::start_capture` blocks (via a `bounded(1)` handshake channel,
≤2s) until the capture thread has actually called `cpal::Stream::play()` successfully, and
returns `Err` on any build/play failure or on timeout rather than handing back a handle that
might not really be capturing. A non-error `CaptureHandle` means the stream really started.

**Consequences:**
- `cpal`, `rubato`, `rtrb` are dependencies again (ADR-010 reversed on this point).
- `crates/vuho-stt-engine` depends on `vuho-audio` for both capture and device
  enumeration/permission — `list_input_devices()`/`request_mic_permission()` (frozen call-site
  signatures) now delegate to `vuho_audio::list_input_device_names()` /
  `vuho_audio::permission`, mapping `AudioError` → `EngineError`.
- The ort static-link gate in `scripts/bundle-macos.sh` (`otool -L … | grep -viE
  '/usr/lib|/System' | grep -q dylib` must find nothing) also covers `vuho-audio`'s
  transitive deps — cpal itself links only system CoreAudio frameworks, no bundled dylib.

**Rejected alternatives:** keeping audio ownership on the Swift side is no longer available as
an option — there is no Swift left in the project (ADR-014). Pushing raw audio across an FFI
boundary into a custom `AudioProcessing` conformer (ADR-001's rejected alternative (a)) is now
moot for the same reason.

### ADR-014 — Native CoreML via `objc2-core-ml` for ASR; `ort` retained solely for VAD

**Status:** Accepted. Supersedes ADR-001 (engine-side mic ownership), ADR-003 (WhisperKit
adapter naming), ADR-005 (Swift build/FFI mechanics).

**Decision:** Speech recognition runs entirely in-process via native CoreML
(`objc2-core-ml`), loading `.mlmodelc` bundles directly with
`MLModel::modelWithContentsOfURL_configuration_error` — no Swift package, no dylib, no
`libloading`, no C ABI boundary of any kind. `.mlmodelc` directories load without a compile
step, including `ParakeetDecoder.mlmodelc`, which has **no `metadata.json`** — its interface is
read from `model.mil` instead, so `CoreMlModel::load` cannot require `metadata.json` to be
present.

**Ground truth: the model set actually shipped** (extracted from the `.mlmodelc` files
themselves — this is the fixed-15-second pipeline variant, **not** FluidAudio's split
`JointDecision` variant, which bakes argmax into the model and returns `token_id`/`duration`
directly):

| Model | Inputs | Outputs |
|---|---|---|
| `Preprocessor.mlmodelc` | `audio_signal` f32 `[1,240000]` (15 s @ 16 kHz, zero-padded), `audio_length` i32 `[1]` | `mel` f32 `[1,128,1501]`, `mel_length` i32 `[1]` |
| `ParakeetEncoder_15s.mlmodelc` | `audio_signal` f32 `[1,128,1501]` (the mel), `length` i32 `[1]` | `encoder_output` f32 `[1,188,1024]`, `encoder_output_length` i32 `[1]` |
| `ParakeetDecoder.mlmodelc` | `targets` i32 `[1,1]`, `target_lengths` i32 `[1]`=1, `h_in`/`c_in` f32 `[2,1,640]` | `decoder_output` f32 `[1,1,640]`, `h_out`/`c_out` f32 `[2,1,640]` |
| `RNNTJoint.mlmodelc` | `encoder_outputs` f32 `[1,1,1024]` (**one** frame), `decoder_outputs` f32 `[1,1,640]` — **no `encoder_length` input**, despite early planning notes assuming one | `logits` f32, 8198 elements |
| `parakeet_v3_vocab.json` | — | 8192 entries, ids 0–8191; blank id 8192 is **not** in the file |

`8198 = 8193 token logits (8192 vocab + blank@8192) + 5 duration logits`; duration = argmax
index directly (no lookup table). `RNNTJoint`'s `logits` output is actually **post-softmax
log-probabilities, not raw logits** — immaterial for greedy argmax decoding (argmax is
invariant under the monotonic log-softmax transform), but worth recording accurately since
"raw logits" was the working assumption during planning.

**Compute-unit assignment is a load-time, per-component decision, not a runtime fallback**
(verified against each bundle's actual `model.mil` signature, not just a summary table):
`Preprocessor` is `CPUOnly` (framing/FFT/mel-filterbank — all CPU ops, matching FluidAudio's
own choice); `ParakeetEncoder_15s` is `CPUAndNeuralEngine` (its inputs are **fixed** shape, so
the ANE can compile one plan — this is the one dispatch decision that actually matters for
performance); `ParakeetDecoder` and `RNNTJoint` are `CPUOnly` because both declare
`RangeDims`-flexible inputs, and loading either with the ANE enabled fails at *prediction*
time with a CoreML E5RT validation error inside the LSTM's `initial_h` shape check (the
compiled ANE plan locks onto the `RangeDims` ceiling instead of the actual per-step shape run
against it). These are small per-frame ops, so CPU-only for them is not a meaningful latency
cost.

**Why `ort`/ONNX Runtime cannot reach the ANE for this workload:** ORT's execution providers
that can dispatch to the Neural Engine need static shapes end-to-end; this model's per-window
mel-frame-count and per-step decoder-target dimensions don't freeze the way CoreML's own
compiler freezes them for the fixed-15s variant, so an ORT path here would fall back to CPU —
the entire reason to prefer native `objc2-core-ml` for the ASR path over `ort`, despite `ort`
already being a dependency (for VAD, see below) and therefore the path of least new surface
area. `ort` is retained **solely for VAD** (the `voice_activity_detector` crate, which embeds
Silero v5 and cannot load external weights), statically linked — verified by the
`scripts/bundle-macos.sh` gate (`otool -L "$APP/Contents/MacOS/vuho" | grep -viE
'/usr/lib|/System' | grep -q dylib` must find nothing). `models/silero-vad/` (fetched fp16
ONNX via `fetch-model.sh`) is provisioned on disk but **not loaded by the app today** — it
exists only as a placeholder for a future direct-`ort` Silero v6 swap, and is deliberately
**not** copied into the `.app` bundle by `bundle-macos.sh`.

**Hard-won lesson: pass `CoreML` model outputs straight through, never extract-and-rebuild.**
The single costliest bug of this rewrite was in `run_encoder`: the `Preprocessor`'s `mel`
output was extracted to a `Vec<f32>` and rebuilt into a fresh `MLMultiArray` before being fed
to the encoder. That round trip silently corrupted the mel content — for reasons never fully
isolated, since the rebuilt array's own strides checked out as standard row-major contiguous —
enough that the joint network only ever saw what looked like near-silence, which for a long
time looked exactly like a defective model or a decode-algorithm bug. It was neither. The
diagnostic method that actually found it was **golden-reference diffing**: an independent
Swift/CoreML script running the same four `.mlmodelc` files standalone, compared frame-for-frame
against the Rust path (`mel_length`, `encoder_output_length`, `encoder_output` mean/max, and
frame-0 joint logits, all to floating-point tolerance). The fix was to stop rebuilding the
array at all — pass the `Preprocessor`'s own `MLMultiArray` straight into the encoder call, as
CoreML's own API is designed for — which sidesteps the question of whether a manual
reserialization correctly accounts for the *model's* returned strides, rather than trying to
answer it. This was the original incident behind the PR4 gate; it is resolved.

**`Send`/`Sync` boundary:** a single documented newtype, `coreml::SendModel`, carries
`unsafe impl Send + Sync` for the loaded model set — justified because Apple documents
`MLModel` prediction as thread-safe, and because every caller in `parakeet/models.rs`
serializes predictions onto one thread per session. No other `unsafe impl Send/Sync` exists in
the engine.

**Consequences:**
- Zero Swift, zero FFI, zero `libloading` anywhere in the project (retires ADR-003's "WhisperKit
  adapter" framing and ADR-005's build mechanics in their entirety).
- `objc2-core-ml`, `objc2-av-foundation`, `block2`, `half` join the `objc2` family already
  mandated by CONSTITUTION rule 7.
- `cargo build` needs no Swift toolchain at all — `build.rs` for `vuho-stt-engine` no longer
  shells out to `swift build`.

### ADR-015 — Sliding-window streaming + token dedup

**Status:** Accepted — batch windowing + dedup **and** the live streaming session are shipped
(the `test-stt-ffi` gate and the non-ignored file-driven streaming test exercise them
end-to-end).

**Decision (shipped — batch path):** Audio is processed in a **15.0 s sliding window**
(`WINDOW_SAMPLES = 240 000` samples @ 16 kHz — exactly 15 s, not 14.96 s) with a **2.0 s
overlap** (`OVERLAP_SAMPLES = 32 000` samples = 25 encoder frames @ 80 ms/frame) and a 13 s
advance per window (`ADVANCE = WINDOW_SAMPLES − OVERLAP_SAMPLES`). The final window is
**end-aligned**: `start = total_len.saturating_sub(WINDOW_SAMPLES)`, floored to a frame
multiple, mirroring FluidAudio's last-chunk warmup behavior, when the remaining audio doesn't
fill a full advance step.

Decoder LSTM state is **fresh per window** (`DecoderState::new()` for every window and for
every partial re-inference of the open window). This deliberately **rejects** the original
plan's carry-across-windows algorithm: FluidAudio's own `ChunkProcessor` decodes chunks in
parallel tasks with fresh state per chunk — cross-window continuity comes entirely from the
2 s **audio** overlap plus seam merging, never from carrying LSTM state. Carrying state was
the root cause of a characterized blank-lock bug (a mid-sentence LSTM prime mismatches the
next window's independently-computed encoder output, and since blank never steps the LSTM,
the mismatch never self-corrects — the window can stay blank for its entire 15 s span).
Within one window's greedy decode loop, the LSTM steps **only on non-blank emissions**.

Overlap dedup (`stream::merge::merge`) matches at **word granularity**, not token IDs: two
independently-decoded windows routinely pick different subword splits for the same word
("▁a"+"sk" vs "▁as"+"k") and disagree on seam-word capitalization/trailing punctuation, so
tokens are grouped into words via the vocab's word-boundary information (`Vocab::piece_info`),
compared case-folded and edge-punctuation-stripped, and the longest contiguous word run inside
the physical overlap region anchors the splice. `merge` returns `MergeOutcome
{ keep_committed, append }` — it may **truncate `committed` at the seam** to prefer the fresh
decode's boundary word, because the committed copy can carry an artifact (e.g. a stray
terminal period from a window that ended mid-phrase) that the fresh decode, with more trailing
context, does not.

**Partial-transcript mapping (design decision, wiring is the in-progress part):**
`DictationEvent::PartialTranscript { current_text, unconfirmed_text }` is produced once by the
STT layer — `current_text` from the committed token sequence, `unconfirmed_text` from the open
window's freshly re-decoded (not yet committed) tokens — so the pipeline and UI never re-derive
the split themselves (CONSTITUTION rule 2). Open item: `vuho-ui/src/overlay.rs`'s
`split_transcript` still re-derives this at the UI layer today from the *batch*-era event shape,
which predates this ADR.

**Resolved — the cross-window continuity gap:** the once-`#[ignore]`d
`crates/vuho-stt-engine/tests/batch_multiwindow.rs` (`jfk.wav` ×3, ≈33 s, three windows) now
runs un-ignored and passes with no dropped or duplicated content at the seams. The fix was
the fresh-per-window decoder state above plus the word-granularity merge — **not**
FluidAudio's `TdtFrameNavigation` `initial_t`/`contextFrameAdjustment` navigation, which an
implementation attempt proved orthogonal to this bug (each batch window computes its own
independent encoder output at local frame 0; the windows overlap in audio, not in a shared
frame-index space).

**Shipped — the live streaming session** (`stream/session.rs`): `start_stream` prechecks mic
permission (`Denied`/`Restricted` → `EngineError::MicPermissionDenied` synchronously), starts
`vuho-audio` capture, and spawns the `"vuho-stt-session"` thread; `stop_stream` takes the
single `SessionHandle` owner, sets the stop flag, joins, and returns the final
`TranscriptionResult` (end-aligned final window over the tail). The loop: accumulates 16 kHz
chunks, VAD-gates re-inference (silence costs zero ANE work), emits `Activity` every ~50 ms
from capture RMS, re-infers the open window at an adaptive ≥1 s cadence from a fresh
`DecoderState`, promotes fresh→committed on a ≥800 ms VAD endpoint without advancing the
window, and commits/advances at the 240 000-sample window boundary. The session loop is a
free function over an `AudioSource` seam and a chunk `Receiver`, so a non-ignored, model-gated
test drives it with `jfk.wav` in 100 ms chunks and asserts a `PartialTranscript` precedes the
final result. The engine emits only `PartialTranscript`/`Activity`/`Error`;
`SessionStarted`/`SessionCompleted` remain the pipeline's to emit.

### ADR-016 — Startup preflight permission gate (replaces reactive/stacked prompting)

**Status:** Accepted.

**Problem:** `make run` (`package` → ad-hoc codesign → `open Vuho.app`) showed **two**
Accessibility dialogs back-to-back even when Accessibility was already granted for Vuho.
Permissions were only discovered reactively, deep inside startup: `wire_production` →
`start_hotkey` → `HotkeyListener::start()` checked `sys::is_accessibility_trusted()` and
returned `Err`; on that `Err`, `permissions::prompt_accessibility()` fired the native
Accessibility dialog *and* an unconditional custom `NSAlert`, stacking both every time. Ad-hoc
signing (`--sign -`) makes this path hit on every single `make run`: each rebuild is a new
binary identity to TCC, so a grant that looks "already allowed" in System Settings doesn't
apply to the freshly-signed binary.

**Decision:** Check every required grant *before* doing any real work. `crates/vuho-ui/src/main.rs`
calls `permission_gate::missing_permissions()` immediately after the accessory activation
policy is set, before the overlay window or `wire_production` are created. If anything is
missing, `permission_gate::open_gate_window` shows **one** window listing exactly what's
missing, with one "Allow …" button per permission, and `main()`'s top-level closure returns —
no overlay, no model warmup, no hotkey start until the user grants everything and relaunches.
When nothing is missing, execution falls straight through to the unchanged production path with
zero dialogs. This also means the ~490 MB model's ANE warmup never starts until every
permission is confirmed, instead of the wasted load that happened today whenever Accessibility
was missing.

**The three checked permissions**, each with a pure (non-prompting) check for the gate's initial
scan and poll loop, plus a fire-and-forget "request" action for the gate's "Allow …" button:
- **Accessibility** — `vuho_os_integration::accessibility_trusted()` (wraps `AXIsProcessTrusted`).
- **Input Monitoring** — `vuho_os_integration::input_monitoring_trusted()`, new: `sys.rs` adds an
  `IOKit` extern block for `IOHIDCheckAccess`/`IOHIDRequestAccess`
  (`kIOHIDRequestTypeListenEvent`). `IOHIDCheckAccess` itself never prompts — only
  `IOHIDRequestAccess` does — which is what keeps the gate's initial all-granted check
  dialog-free.
- **Microphone** — `vuho_stt_engine::mic_permission_granted()`, a new pure sibling to the
  existing `request_mic_permission()` (which has the side effect of firing the native prompt on
  `NotDetermined`); delegates to `vuho_audio::mic_authorization_status()`.

**Self-relaunch requirement:** `CGEventTapCreate`'s documented behavior is that a live
Accessibility grant doesn't retroactively arm an already-created tap — a fresh process is
required (see the existing comment on `prompt_accessibility_trust`). Once every permission is
granted, the gate window's poll loop swaps its content for a single "Relaunch Vuho" button,
which re-execs `std::env::current_exe()` via `std::process::Command::spawn()` then
`std::process::exit(0)` — only exiting if the spawn actually succeeded, so a failed relaunch
doesn't strand the user with no window at all. This works identically for `cargo run`'s raw
binary and the packaged `.app`'s binary.

**Empirical deviation from the original plan:** the plan's straw-man for `HotkeyListener::start()`
was to hard-fail (`Err(OsError::Hotkey)`) whenever *either* Accessibility *or* Input Monitoring
was missing. A throwaway standalone probe binary calling
`IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)` on the development machine — a machine whose
already-installed, ad-hoc-signed `Vuho.app` has Accessibility granted and a working CapsLock
hotkey, but where Input Monitoring had never been explicitly granted to anything — read `Denied`.
(TCC grants are scoped per code signature; a fresh, never-launched probe binary cannot directly
read another process's grant, and reading the system TCC database to check `Vuho.app`'s own
entry directly was not permitted in this environment — but a `Denied` result from *any* freshly
built, never-granted binary on this machine is the same result a freshly rebuilt `Vuho.app`
would get today, given ad-hoc signing resets grants on every rebuild, which is precisely the
scenario this ADR's gate exists to handle cleanly.) Hard-failing `start()` on that condition
would have regressed a setup that works today into a permanent gate. `HotkeyListener::start()` therefore
keeps Accessibility as its only hard gate; a missing Input Monitoring grant is logged as a
warning, not a hard failure. The preflight gate window still lists and offers to request Input
Monitoring regardless (all three permissions are always checked and offered there) — only the
hotkey listener's own hard-fail condition is narrowed.

**Consequences:**
- `permissions::prompt_accessibility()` drops its redundant `NSAlert` entirely (now just
  `vuho_os_integration::prompt_accessibility_trust()`), and the now-unused
  `ACCESSIBILITY_SETTINGS_URL` constant is removed with it. It remains as a defensive fallback
  for the two reactive call sites where a grant is revoked **mid-session**
  (`start_hotkey`'s error branch, `settings_window.rs`'s `select_hotkey` error branch) — a single
  native dialog (or none, if still trusted) there too.
- Ad-hoc rebuild signatures still reset TCC grants on every `make run` in local dev — the gate
  doesn't remove that cost, it makes each reset a single clean "one window, one click per
  permission, one relaunch" flow instead of stacked dialogs. `SIGN_ID` (`scripts/package.sh`)
  remains the only way to avoid the re-grant entirely across rebuilds.
- `--features demo` skips the gate entirely (cfg-gated at the `mod permission_gate;` declaration
  and the call site in `main()`), matching demo mode's existing `#[cfg(not(feature = "demo"))]`
  gating of hotkey/permission code — demo mode has no mic/hotkey/permissions involved.

**Denied-state handling (amendment):** the original gate's "Allow …" button always called each
permission's native `request()` prompt. macOS only actually shows that prompt when the grant is
still *NotDetermined* — once a grant is explicitly **Denied** (the user said no, or an MDM policy
blocked it), `IOHIDRequestAccess`/`request_mic_access_async` are silent no-ops and the user was
stuck in the gate with a button that visibly did nothing. The gate now models each permission's OS
status as a tri-state `Access { Granted, Promptable, Denied }` (CONSTITUTION rule 2 — data, not an
inferred bool) instead of `is_granted() -> bool`:
- **Microphone** (`AVFoundation`) and **Input Monitoring** (`IOHIDCheckAccess`, which already
  returns a tri-state `IOHIDAccessType` that the original bool wrapper threw away) both expose
  their real three-way status, mapped to `Access` by pure, unit-tested functions
  (`mic_access`, `input_monitoring_to_access` in `permission_gate.rs`).
- **Accessibility** is an OS-level asymmetry, not a shortcut taken here: `AXIsProcessTrusted` is a
  plain bool with no NotDetermined-vs-Denied distinction, so `accessibility_access` can only ever
  report `Granted` or `Promptable`. Its "Allow Accessibility" button therefore always re-fires
  `AXIsProcessTrustedWithOptions`, whose own native dialog already carries an "Open System
  Settings…" button when the grant was previously denied — the OS handles that case natively for
  this one permission.
- A row showing `Access::Denied` (Microphone/Input Monitoring only) renders an "Open System
  Settings" button instead of "Allow …", deep-linking straight to that permission's pane via
  `NSWorkspace` (`permissions::open_url`, made `pub(crate)` and reused rather than duplicated —
  CONSTITUTION rule 26). All three settings-pane URLs live in one place, `permissions.rs`:
  `MICROPHONE_SETTINGS_URL` (pre-existing, also used by `show_microphone_denied`),
  `ACCESSIBILITY_SETTINGS_URL` (re-added — it was removed as unused when this ADR first shipped,
  since nothing consumed it until this amendment), and `INPUT_MONITORING_SETTINGS_URL` (new,
  `Privacy_ListenEvent` — the Input Monitoring TCC service key found in
  `Security.prefPane`'s `PrivacyTCCServices.plist`; there is no scriptable UI element to derive
  this anchor from the way Microphone/Accessibility's anchors can be, so it is confirmed against
  public macOS deep-link references rather than driven end-to-end in this environment).

### ADR-017 — LLM-based transcript cleanup (Candle + LFM2.5-1.2B, Q6_K, Metal)

**Status:** Reverted. The LLM cleanup experiment was abandoned due to poor quality — hallucinations,
over-aggressive paraphrasing, and translation of unsupported languages. The rule-based
`vuho-postprocess` crate was restored in its place.

**History:** ADR-017 was originally accepted as a deliberate reversal of the project's
"no LLM" principle (`CLAUDE.md` point 3). It fully replaced the rule-based `vuho-postprocess`
crate with an on-device LLM — **LFM2.5-1.2B-Instruct** (`LiquidAI/LFM2.5-1.2B-Instruct-GGUF`,
quantized **Q6_K**, ~918 MiB) — via **Candle** (`candle-core`/`candle-transformers`'s
`quantized_lfm2` module, pure Rust, Metal backend). The `TextCleaner` trait lived in
`vuho-domain`, `vuho-cleanup` owned Candle, and a `CleanupWorker` thread ran post-process
off the command thread (ADR-017 / Bug 2 fix). The cleanup model was provisioned by
`fetch-model.sh lfm2` and bundled into `Contents/Resources/` at package time.

**Reversal rationale:** The LLM produced unacceptable output — hallucinating content, translating
unsupported languages instead of passing them through, and paraphrasing rather than cleaning.
The rule-based `vuho-postprocess` (filler removal, spacing normalization, newline collapse) was
restored. The `vuho-cleanup` crate, `TextCleaner` trait, `CleanupWorker` thread, `Processing`
event, and all Candle/CleanupModel dependencies were removed. The pipeline's `emit_result`
now calls `vuho_postprocess::postprocess` inline, the same thread that performs clipboard
write + ⌘V injection.

---

### ADR-018 — `PartialTranscript` carries producer-supplied `confirmed_text`/`unconfirmed_text`

**Status:** Accepted. Closes CONSTITUTION rule 2 ("no re-derivation of producer state in a
consumer") against a concrete violation that existed in the domain event shape itself, not just
in one call site.

**Decision:** `DictationEvent::PartialTranscript` carries two producer-owned `String` fields,
`confirmed_text` and `unconfirmed_text`, replacing the old two-field shape
`{ current_text, unconfirmed_text }` where `current_text` was the *whole* transcript
(confirmed + unconfirmed concatenated) and the UI had to reconstruct the confirmed-only prefix
itself via `current_text.strip_suffix(&unconfirmed_text)` — a re-derivation that silently broke
whenever the unconfirmed tail wasn't a literal suffix of the whole string (e.g. after the
seam-merge logic in ADR-015 rewrites a boundary word). `vuho-stt-engine`'s streaming
`Accumulator` (`stream/session.rs`) already computes both halves internally to drive the VAD/seam
logic; this ADR just stops throwing that split away before it reaches the domain event.

**Consequences:**
- `vuho-ui`'s `OverlayModel::handle_event` assigns both fields verbatim
  (`self.confirmed_text = SharedString::from(confirmed_text)`, same for `unconfirmed_text`) — zero
  string slicing, zero suffix-stripping, in the overlay. Falsifiable by grep: no `strip_suffix` or
  manual byte-index slicing remains anywhere in `vuho-ui/src` touching transcript text.
- Every event-construction call site across the workspace (`vuho-stt-engine`'s partial/commit/final
  emit points, test fixtures in `vuho-dictation`/`vuho-ui`) was updated to supply both fields
  explicitly — there is no default or derived value; a caller that doesn't know its confirmed text
  cannot construct the event with only the whole string.
- `DictationEvent`, `ErrorKind`, and `InjectionOutcome` all gained `#[non_exhaustive]` as part of
  the same domain-hardening pass (WP6) — later found (final review) to have the opposite of the
  intended effect: `#[non_exhaustive]` is for enums whose variants a crate's *external* consumers
  cannot exhaustively enumerate, and it works by *forcing* every downstream `match` to carry a
  wildcard arm, so a variant this crate adds later is silently caught by that wildcard instead of
  producing a compile error. Every consumer of these three enums lives inside this workspace, so
  the attribute was removed: an exhaustive `match` with no wildcard arm (`vuho-ui`'s `event_loop.rs`
  and `overlay.rs`) is what actually makes a future variant a compile error at every call site,
  which is what this consequence always intended to describe.

**Rejected alternatives:** keeping `current_text` as a third, redundant field alongside the two
producer-supplied halves (concatenation is trivial for a consumer that wants the whole string,
and a third field that must always agree with the other two is exactly the kind of duplicated,
driftable state CONSTITUTION rule 2 exists to eliminate) — rejected unless a real consumer needs
the concatenation, which none currently does.

---

### ADR-019 — Model manifest: one file, one crate, is the chokepoint for model identity + paths

**Status:** Accepted. Closes CONSTITUTION rule 26 ("one chokepoint, not two near-identical
copies") against a concrete instance: the Parakeet-TDT model name, its `CoreML` component list,
the cleanup model's name/assets, and the bundle ID (`tech.iurii.vuho`) were each duplicated
across `vuho-stt-engine`, `vuho-cleanup`, and four provisioning/packaging shell scripts
(`fetch-model.sh`, `bundle-macos.sh`, `package.sh`, `verify-app.sh`), kept in sync only by
"MUST match" comments a human had to notice and honor by hand. *(Corrected below: `vuho-cleanup`
was deleted when ADR-017's LLM cleanup experiment was reverted, so the `CleanupManifest`
accessor this ADR originally added no longer exists, and `resolve_model_folder`'s sole caller
today is `vuho-stt-engine`. The chokepoint decision this ADR records is unaffected — it just has
one caller instead of two now.)*

**Decision:** `models.manifest.json` (repo root) is the single source of truth for the
Parakeet-TDT model's upstream repo, pinned revision, component/asset file lists (asset roles are
**named fields** — not positional array entries, so no consumer can silently couple to array
ordering), directory name, override env-var names, and the macOS bundle ID; it also carries the
Silero VAD provisioning metadata `fetch-model.sh` uses. A new std-only crate, `vuho-model-paths`,
`include_str!`s the manifest at compile time, exposes typed accessors (`Manifest`, `SttManifest`,
`SileroManifest`, `ModelSpec`), and owns `resolve_model_folder` — the **one** copy of the env-var
→ `.app` bundle `Contents/Resources` → workspace-relative `models/` fallback chain. At the time
this ADR was accepted, that resolver replaced near-verbatim duplicates in
`vuho-stt-engine::resolve_model_folder` and `vuho-cleanup::resolve_cleanup_model_folder`
(including their private `non_empty_env`/`bundle_resources_dir`/`dev_model_dir` helpers); the
`vuho-cleanup` side no longer exists (see the status note above). The parsed manifest is cached
in a `OnceLock<Result<Manifest, serde_json::Error>>` — the `Result` is stored, not just the `Ok`
value, so a malformed manifest can never poison the lock into a permanently-unusable state
(CONSTITUTION rule 17); the caller-visible panic happens on every access via `.expect(...)`, not
just the first.

The four shell scripts read the same `models.manifest.json` at runtime through a new shared
`scripts/manifest-lib.sh` (`python3 -c` one-liners against the manifest, the same technique
`fetch-model.sh` already used before this ADR) instead of each hardcoding the same strings —
renaming a component in the manifest now breaks the Rust build (a resolver that can't find the
renamed file) **and** every script that provisions/packages/verifies it, loudly, in the same
place a human would look first.

**Consequences:**
- `vuho-stt-engine` depends on `vuho-model-paths`; its own `resolve_model_folder` function is a
  thin wrapper that builds a `ModelSpec` from the manifest and delegates. *(`vuho-cleanup` and
  its `resolve_cleanup_model_folder` no longer exist, per the status note above — at the time
  this ADR was accepted there were two such callers.)*
- Zero model names, revisions, component lists, or the bundle ID remain as literals anywhere in
  Rust source or the four scripts outside `models.manifest.json` itself (WP12's grep-falsifiable
  check).
- `vuho-model-paths` is deliberately std-only (`serde`/`serde_json` only) so it builds on any
  host, keeping the macOS-only boundary exactly at its two callers, not widening it.

**Rejected alternatives:** a build-time code-generation step (`build.rs` emitting Rust constants
from the manifest) — rejected as unnecessary indirection; `include_str!` + `serde_json` at
first-use is simpler, has no build-script maintenance burden, and the manifest is small enough
that runtime parsing cost is immaterial (measured: parsed once per process via the `OnceLock`,
not per call).

### ADR-020 — The model is a user resource, provisioned once, verified against a repo-pinned lock

**Status:** Accepted. Supersedes ADR-008.

**Problem:** ADR-008's "bundle the model, no download ever" pins the Parakeet-TDT CoreML model
(~474 MB) inside `Contents/Resources/`, so `Vuho.app` is 504 MB. That is incompatible with
Homebrew cask distribution — a cask artifact is expected to be a thin download, not half a
gigabyte, and users installing via `brew install --cask` should not pay for a model they may
never launch the app to use. There are no Vuho releases today, and this bundle size is why: a
504 MB GitHub release asset for a private, low-traffic app is the wrong shape for the distribution
path the project actually wants (`brew install --cask vuho`).

**Decision:** The `.app` becomes model-optional. `vuho_model_paths::resolve_model_folder` gains a
fourth, lowest-priority candidate — `~/Library/Application Support/Vuho/models/<name>`
(`user_models_dir()`, the single definition of the download location) — after the existing
`$VUHO_MODEL_FOLDER` → `.app` bundle `Contents/Resources/<name>` → workspace `models/<name>`
chain. A new crate, `vuho-model-fetch`, downloads the model into that directory (`hf-hub` 1.0,
Xet-first with automatic HTTPS fallback, driven by an explicit user action — never automatically)
and verifies it against `models.lock.json`, a generated file committed to the repo alongside the
hand-edited `models.manifest.json` (ADR-019): the manifest still says *what* model and *which*
components; the lock adds the per-file sizes and SHA-256 hashes needed to tell a complete,
untampered download apart from a partial or corrupted one.

**The central invariant.** This is the load-bearing part of this ADR: **verification applies
only to bytes Vuho fetched over the network.** Out-of-band provisioning keeps ADR-008's existing
trust model exactly as it was — this ADR does not make Vuho suspicious of a model it didn't
download itself.

- `resolve_model_folder()` stays authoritative and is always called first.
- If the resolved path is **not** under `user_models_dir()` — the env override, the `.app`
  bundle's `Contents/Resources/`, or the workspace `models/` dev directory — the model is
  `Ready` with no sidecar check at all. Those three trees are provisioned out-of-band
  (`scripts/fetch-model.sh`, `scripts/bundle-macos.sh`) and trusted exactly as ADR-008 always
  trusted them.
- Only the one tree Vuho itself downloaded — under `user_models_dir()` — is verified against the
  lock before being declared `Ready`.

A sidecar-gated check applied uniformly to all four candidates would be wrong, not merely
redundant: `scripts/bundle-macos.sh` copies the model into `Contents/Resources/` with a bare
`cp -R` and writes no sidecar, and `scripts/fetch-model.sh` writes into `models/` the same way.
A uniform gate would therefore report `Missing` for the DMG build, for `cargo run`, for
`VUHO_MODEL_FOLDER`, and for `test-stt-ffi` — i.e. every path except the one this ADR actually
adds — and offer to re-download 474 MB on top of a model already sitting on disk. Scoping
verification to `user_models_dir()` is what keeps those paths behaving exactly as before.

**Consequences:**
- Two distribution shapes come out of one build script (`scripts/bundle-macos.sh`,
  `VUHO_BUNDLE_MODEL` flag): a DMG with the model embedded — offline from first launch, still
  fully supported — and a Homebrew cask without it — measured ≈40 MB on disk (≈15 MB as the
  gzipped release tarball; grows with the binary's own dependencies, so treat this as an order of
  magnitude, not a promise), first-run download. Neither
  is a new build system; both are the same script with the model-copy step made conditional.
- `vuho-model-fetch` is the **only** crate in the workspace permitted to perform network I/O —
  stated here as an invariant a future review can check by grepping the dependency graph for
  HTTP/TLS crates outside it, the same way ADR-019's "zero literals outside the manifest" is
  grep-falsifiable.
- ADR-012's error recovery matrix gains a real model-download row (see the fix to that ADR
  below) in place of the "N/A" it carried while the model could never fail to be present.
- Ad-hoc signing (no Apple Developer ID exists for this project) means every release is a new
  code identity to TCC, so users re-grant Microphone / Accessibility / Input Monitoring on every
  upgrade, cask or DMG. This is accepted and documented in the cask's `caveats`, not solved by
  this ADR — the durable fix (Developer ID + notarization) is out of scope.
- ADR-016's startup preflight gate generalizes into a readiness window that can show either a
  missing permission or a missing model. The two entry paths stay deliberately distinct: the
  model row is **never** shown on the pre-`wire_production` permission-gate path, because that
  path runs before `VuhoState` exists, and the Download button has nothing to send its command
  to without `VuhoState`'s command sender. A model-missing state is discovered only after
  `wire_production` runs, and is handled by the same readiness window through a second, separate
  entry point instead.

**Enforcement:** `vuho_model_paths::resolve_model_folder` remains the one chokepoint that decides
where the model lives (ADR-019, unchanged by this ADR); `vuho_model_fetch::availability()` is the
one chokepoint that decides whether the resolved path is trustworthy, and it is a thin layer on
top of the resolver, not a parallel path — it calls `resolve_model_folder()` first and only
inspects the lock when the resolved path is under `user_models_dir()`. `ParakeetEngine::load`
still takes an already-resolved folder (ADR-008's original enforcement point), so an engine
loaded from a `Failed` or `Downloading` model state remains structurally impossible: nothing
calls `ParakeetEngine::load` until `availability()` reports `Ready`.

**Rejected alternatives:** gating engine load on a sidecar-and-lock check for every resolved
path, regardless of which candidate produced it — the design this ADR explicitly rejects (see
"the central invariant" above): it would report `Missing` for every out-of-band-provisioned
path, which is the DMG build, `cargo run`, `VUHO_MODEL_FOLDER`, and `test-stt-ffi` — i.e.
everything except the one new path this ADR adds.

### ADR-021 — Single-panel UI (two presentations)

**Status:** Accepted. Supersedes the three-window design; amends ADR-016 and ADR-020.

**Problem:** the UI had accreted three independent `NSWindow`s with no shared visual language or
lifecycle: the dictation overlay (`overlay.rs`, always present but hidden, `WindowKind::PopUp`,
click-through), a lazily-created settings window (`settings_window.rs`, `WindowKind::Normal`,
opened via `Cmd+Shift+S` or a status-bar menu item), and a separate permission/model readiness
window (`readiness.rs`, also `WindowKind::Normal`, opened either as ADR-016's startup gate or
ADR-020's model-download prompt). Each window duplicated construction boilerplate (dropdown
widgets, button styling, centering math), and the readiness window in particular existed in two
subtly different modes (`ReadinessMode::Gate`/`Production`) with their own poll loops, dismissal
tracking, and reopen logic. The status bar itself had two installs (`install`/`install_gate`) and
two delegate modes to match. None of the three windows shared a design system — `theme.rs` only
ever styled the overlay.

**Decision:** one window, `panel::PanelRoot`, in **one constant frame**, with two presentations
painted into it:
- **Hud** — the dictation overlay: click-through, no keyboard focus, shown on `SessionStarted`,
  auto-hidden per outcome duration (`overlay::outcome_hide_delay`). It paints only the bottom
  `overlay::HUD_CHROME_HEIGHT` (180px) band of the frame and leaves the rest transparent, so it
  looks and sits exactly like the old standalone overlay window. Semi-transparent by design
  (`overlay::PANEL_BG_OPACITY` = 0.65 over `theme::PANEL_LIGHTNESS` = 0.12) — the desktop stays
  visible behind it, per the product spec's "semi-transparent overlay".
- **Full** — a near-opaque (alpha 0.97), tabbed presentation (Overlay / Settings) filling the whole
  frame, replacing both the old settings window and the readiness window. It stays near-opaque
  where the Hud is see-through because it is a surface the user reads and clicks. The Settings tab
  (`crate::settings_tab::SettingsTab`, built in WP5 but never wired to anything user-reachable
  until this ADR) shows permission rows, speech-model provisioning, and the microphone/hotkey
  dropdowns in one scrollable column, reading `StatusModel` for every piece of live state instead
  of re-deriving it.

The window itself is created once, at startup, in `Presentation::Hud`, `shown: false` — exactly
like the old overlay was. Its frame comes from one geometry source, `panel::panel_bounds`: 460×480
horizontally centered, 120px above the bottom of the primary display (clamped to the display's top
edge on a display too short to hold it), used both at creation and on every presentation switch.
The two presentations originally had their own geometries — 460×180 bottom-center for the Hud,
460×480 screen-centered for the Full — and switching between them moved *and* resized the window
under the user, which read as two different windows rather than one; sharing a frame is what makes
"one window" true on screen and not just in the process.

Switching presentation is one **surgery chokepoint** (`panel::apply_presentation`): it re-applies
that same frame (not redundant — it re-resolves against whichever display is primary *now*, so the
panel follows a display change since it was last shown), then `Hud` turns click-through on, while
`Full` turns click-through off and — as long as a dictation session isn't currently recording —
calls `makeKeyAndOrderFront:`
(`window_config::make_key_and_order_front`), giving the panel keyboard focus *without* activating
the application, since GPUI's `WindowKind::PopUp` already produces a non-activating `NSPanel`
(`NSWindowStyleMaskNonactivatingPanel`) that can become key on its own. While recording, `Full`
instead calls plain `orderFront:` and never grabs key status at all (`show_full`'s `grab_key`
parameter, derived from `OverlayModel::is_recording`) — opening the panel mid-dictation (a
`Failed` model status, a tray click) must not steal the destination of `inject_text`'s synthesized
⌘V from the app the user is actually dictating into. The window's level stays
`kCGScreenSaverWindowLevel` (1000, set once at creation) across both presentations, so the Full
presentation floats above normal windows exactly like the Hud does.

Three further transitions build on that chokepoint: `show_full` (tray click, `Cmd+,`, a `Failed`
model status), `on_session_started` (a session beginning while the panel is hidden shows the Hud;
while the Full presentation is already open, it only switches the active tab back to Overlay, but
if that window happens to be key it
resigns key status via `window_config::resign_key_keep_front` — `orderOut:` then plain
`orderFront:` — for the same reason `show_full` withholds it above: a session starting means the
panel is about to lose the ⌘V destination race to whatever app the user is dictating into), and
`hide_if_hud` (the old outcome-duration auto-hide, now a no-op while the Full presentation is
open). A finished session whose outcome still needs attention (`ClipboardOnly`/`Failed`) re-shows
the panel as the Hud even if it was already dismissed mid-session
(`event_loop::maybe_show_hud_for_outcome` → `panel::show_hud_for_outcome`, sharing the same
"show as Hud if not already shown" step `on_session_started` uses).

**Dismissal affordances**, all going through the one `hide`/`hide_root` implementation: the Full
presentation's tab-strip "✕" button, `Esc` (bound to the `ClosePanel` action on both the
permissions-blocked and production startup paths), and clicking an already-open tray icon a second
time (`open_from_tray`'s toggle). `hide_root` also closes any open Settings dropdowns
(`SettingsTab::close_dropdowns`) so a dismiss-then-reopen never shows a stale device list.

**The permission gate is now the panel's Settings tab (amends ADR-016).** `main.rs` builds the
settings store, the `StatusModel`/`SettingsTab` entities, and the panel itself *before* checking
`readiness::missing_permissions()`. If anything is missing, `wiring::wire_production` never runs
this launch — instead the panel opens on its Settings tab
(`panel::show_full(panel, Tab::Settings, cx)`), and the tray installs with `cmd_tx: None` (the
toggle item is a no-op; `CompositeStatus::toggle_enabled` also disables it). The permission rows
and, once every grant lands, the relaunch row (`CompositeStatus::RelaunchRequired`) live in
exactly the same Settings tab a fully-launched app's Cmd+, would open — there is no separate gate
window, and no separate `install_gate`/`DelegateMode::Gate`/`GateCommand` machinery in
`status_bar.rs`: one `install(cmd_tx: Option<Sender<DictationCommand>>, ui_tx)` call serves both
states, with `sync`'s existing `CompositeStatus::menu_title`/`toggle_enabled` already covering
`PermissionsMissing`/`RelaunchRequired`. A permissions poll (`panel::start_permissions_poll`)
replaces the old readiness window's `spawn_poll_loop`, writing `StatusModel::permissions_missing`
only when it actually changed. Its lifecycle is **not** simply tied to the panel's own visibility:
`show_full` spawns it (guarded against a duplicate spawn if one is already running) and seeds
`permissions_missing` synchronously first, so the Settings tab's very first paint is already
truthful instead of waiting on the poll's own first tick; the poll then keeps re-checking every
500 ms and self-terminates only once `!shown && permissions_missing.is_empty()` — i.e. the panel
is hidden *and* every permission has actually been granted. This keep-alive-while-non-empty rule
exists because the poll is the only thing that ever clears `permissions_missing`: if it stopped
the moment the panel was merely hidden, a permission granted after dismissal would leave the
tray/menu stuck reporting "Permissions…" forever, with nothing left running to notice the grant.
`hide_root` mirrors the same condition — it drops the task early only when permissions are already
empty, otherwise leaving it to keep converging past the dismiss.

**ADR-020's "model row never on the gate path" invariant is now enforced by a type, not a
window-selection branch.** The old readiness window kept the model row off the permission-gate
path by simply never calling `handle_model_status` from that entry point. The panel enforces the
same fact structurally: `StatusModel.model: Option<ModelStatus>` starts (and, on the
permissions-blocked path, stays) `None` for the process lifetime — `wire_production`, the only
code that ever calls `ui_tx.send(UiCommand::ModelStatus(..))` (via `wiring::send_phase_status`),
never runs on that path — and every render site that would show a model row
(`SettingsTab::render`'s `should_show_speech_model_section`, `PanelRoot::render_idle_status`'s
`CompositeStatus::ModelMissing`/`Downloading`/`Verifying` arms) is downstream of that same
`Option`. A model row appearing on the gate path is therefore not a reachable state, not merely an
unexercised one.

**Consequences:**
- `settings_window.rs` is deleted outright (`SettingsView`, `open_settings_window`, the
  `VuhoState::settings_window` singleton field). `readiness.rs` sheds every window/render/poll
  item (`ReadinessView`, `ReadinessMode`, `open_permission_gate_window`,
  `reopen_or_front_gate_window`, `reopen_or_front_production_window`, `handle_model_status`,
  `spawn_poll_loop`, `spawn_gate_command_drain`, `GateCommand`, every `render_*`/`*_button`
  helper) and keeps only the data model both remaining callers need: `Permission`/`Access`/
  `missing_permissions`/`model_status_text`/`format_mb`/`relaunch`.
- `VuhoState` (the process-lifetime `Global` holding the settings store, hotkey listener, command
  channel, and settings-window handle) is deleted entirely — nothing needs a global anymore.
  `SettingsTab` owns the hotkey listener/command-sender it needs directly
  (`SettingsTab::connect_hotkey`, called once by `wiring::wire_production` after
  `wiring::start_hotkey` succeeds), and the Download/Retry button's `Sender<ProvisionCommand>` is
  constructor-injected into `SettingsTab::new` by `main.rs`, never reached through a global.
- `app_state::UiCommand` sheds `OpenSettings`/`OpenReadiness` (both already unreachable after
  WP4's tray click-split); `OpenPanel` now resolves to `panel::open_from_tray`, which shows the
  Overlay tab while a session has live content and whichever tab was last active otherwise.
- `overlay.rs`'s chrome and content are split (`overlay::hud_chrome` wraps
  `OverlayModel::render_content` for the Hud arm; the Full presentation's Overlay tab embeds
  `render_content` directly, inside the panel's own opaque chrome, never the Hud's translucent
  one) — zero behavior change to the Hud presentation itself, verified by moving
  `bottom_center_origin`'s two existing unit tests into `panel.rs` unmodified.
- `theme.rs` (previously overlay-only, with a blanket `dead_code` allow for the tokens
  `settings_tab.rs`/`controls.rs` were already coded against) is now the shared token set for
  every rendered surface across both presentations; the allow stays because roughly a dozen of
  those tokens are consumed only by production-only (`#[cfg(not(feature = "demo"))]`) modules and
  are therefore genuinely unreachable in a `--features demo` build — a structural fact of a
  cfg-split consumer set, not unfinished wiring.

**Rejected alternative:** keeping the readiness window's two `ReadinessMode`s as-is and merely
retargeting the Settings tab to open it in the appropriate mode — rejected because it would have
preserved two independent "let the user finish setup" implementations
(`readiness.rs`'s row-building + `settings_tab.rs`'s, already written and better factored against
`StatusModel`) with no way for a future edit to keep both in sync short of reading both files
every time. Deleting the older one and keeping the single reads-`StatusModel` implementation is
the one-source-of-truth outcome CONSTITUTION rule 26 asks for.

---

## Target architecture (after these ADRs)

**Crates (eleven total — `vuho-audio` reinstated, ADR-013; `vuho-model-paths` added, ADR-019;
`vuho-cleanup` deleted when ADR-017's LLM cleanup experiment was reverted; `vuho-model-fetch`
added, ADR-020; no Swift package, ADR-014):**
- `vuho-domain` — types/events/commands, no deps, no platform code. `DictationCommand::Toggle`;
  `PartialTranscript { confirmed_text, unconfirmed_text }` (both producer-supplied, ADR-018);
  `TranscriptSegment`/`TranscriptionResult`; `ModelStatus` (ADR-020).
- `vuho-audio` *(reinstated, ADR-013)* — `cpal` capture thread owning the `!Send` `Stream`, `rtrb`
  ring buffer, `rubato` resample to 16 kHz mono, device enumeration, `AVCaptureDevice` mic
  permission (`objc2-av-foundation`). No `vuho-*` dependencies — a leaf crate the engine consumes.
- `vuho-stt-engine` — `TranscriptionEngine` **trait** + `ParakeetEngine` (native CoreML,
  ADR-014): loads the four Parakeet-TDT `.mlmodelc` components + vocab, runs greedy TDT decode
  over a 15 s sliding window (batch, shipped) or a live growing buffer (streaming, **in
  progress** — ADR-015). Also owns the Silero VAD wrapper (`vad.rs`, `voice_activity_detector`).
- `vuho-dictation` — session state machine: `Toggle`/`Start`/`Stop` → `start_or_stop`; wires
  engine events → `DictationEvent`s; on stop → cleanup → inject (`handle_stop` +
  `emit_result`, split per CONSTITUTION rule 28).
- `vuho-postprocess` — rule-based text post-processing: filler removal, spacing normalization,
  newline collapse. No external deps beyond `vuho-domain`.
- `vuho-os-integration` — `objc2` impls: CapsLock/chord hotkey (`CGEventTap`), `inject_text`
  (`NSPasteboard` + `CGEvent` Cmd→V), TIS language detection, clipboard. No `arboard` anywhere.
  Settings-free — the `HotkeySetting` → hotkey-config mapping lives in `vuho-ui`.
- `vuho-settings` — serde-only persistence: `HotkeySetting` preset + microphone device name,
  atomic load/save to `~/.config/vuho/settings.json`.
- `vuho-model-paths` *(new, ADR-019)* — std-only chokepoint crate: embeds `models.manifest.json`
  at compile time, exposes typed manifest accessors and the single `resolve_model_folder`
  env-var → bundle → workspace-dev → user-data (ADR-020) resolution chain, plus the embedded
  `models.lock.json` accessors and the shared `atomic_write` helper. No macOS-specific
  dependencies.
- `vuho-model-fetch` *(new, ADR-020)* — the **only** crate in the workspace permitted to perform
  network I/O: `availability() -> ModelStatus` (the sidecar-and-lock verification chokepoint,
  scoped to the user-data candidate only) and `download()` (hf-hub 1.0, Xet-first with automatic
  HTTPS fallback; clears any leftover `<dir>.partial` → fetches into a fresh `<dir>.partial` →
  fully verifies it → writes the sidecar **inside** `<dir>.partial`, only once verification has
  passed → atomically renames `<dir>.partial` to `<dir>`, promoting the verified bytes and their
  sidecar together). Depends on `vuho-model-paths` and `vuho-domain`.
- `vuho-ui` — GPUI: a single non-activating panel (`panel::PanelRoot`, ADR-021) with two
  presentations — Hud (the dictation overlay) and Full (a tabbed Overlay/Settings window; the
  Settings tab holds the mic + hotkey-preset dropdowns, save-on-change, live hotkey rebind, and
  the ADR-016/ADR-020 permission + model provisioning rows); produces the `vuho` binary.
  Status-bar menu, quit hotkey `Cmd+Option+Shift+Q`, `Cmd+,` opens the panel on Settings
  (`LSUIElement`, no Dock icon).
- `test-stt-ffi` — batch STT regression binary; the deterministic `PASS`-on-`jfk.wav` gate.

**No Swift engine package** — the row this table used to have for one is gone; there is no
Swift anywhere in the project (ADR-014).

**Streaming data flow (target — batch path below is shipped; the streaming half is marked
in-progress per ADR-015):**
```
CapsLock tap (vuho-os-integration, objc2 CGEventTap)
  → DictationCommand::Toggle → DictationPipeline (start)
    → detect_language() [TIS] → engine.start_stream(language, device)
      → [IN PROGRESS] vuho-audio::start_capture (cpal thread, 16 kHz mono chunks)
        → session thread: VAD-gated windowed re-inference (ADR-015) → crossbeam
          → DictationEvent::PartialTranscript / Activity{level: real RMS}
            → GPUI overlay: live text + green light + waveform (real audio energy, ADR-013)
CapsLock tap again → DictationPipeline (stop)
  → engine.stop_stream → final TranscriptionResult
    → vuho_postprocess::postprocess(text, language) [rule-based: filler removal, spacing,
      newline collapse] → os-integration::inject_text
      (NSPasteboard + CGEvent Cmd→V) → lands in the previously-focused app (overlay never took focus)
```
**Batch path (shipped, exercised by `test-stt-ffi`):**
```
jfk.wav → ParakeetEngine::transcribe(samples, language)
  → windower::plan (15 s windows, 2 s overlap, 13 s advance, end-aligned final window)
    → per window: Preprocessor → ParakeetEncoder_15s → tdt_greedy (Decoder + Joint, LSTM
      state threaded across the whole session) → stream::merge (overlap dedup)
  → TranscriptionResult { segments, full_text, language }
```

## Implementation roadmap

Phases 0–4 below (rename spike, streaming engine, OS integration, pipeline, overlay PoC,
packaging) were the original WhisperKit-era sequencing and are superseded in substance by the
PR1–PR6 sequencing that actually shipped the Parakeet-TDT rewrite:

0. ~~Rename spike + build automation~~ — moot; there was nothing left to rename once WhisperKit
   was retired wholesale rather than renamed (ADR-003/005 Superseded).
1. **Teardown + provisioning** (done) — WhisperKit/Swift/FFI removed entirely;
   `scripts/fetch-model.sh` provisions the Parakeet-TDT + Silero VAD models. (ADR-008/014)
2. **`vuho-audio` + VAD + deps** (done) — cpal capture thread, rubato resample, `rtrb` ring,
   `AVCaptureDevice` permission; `voice_activity_detector` wrapper. (ADR-013)
3. **CoreML batch engine** (done) — `ParakeetEngine::transcribe` via `objc2-core-ml`; the
   `test-stt-ffi` `PASS` gate is green. (ADR-014)
4. **Streaming orchestrator + UI parity** (done) — live `start_stream`/`stop_stream`,
   VAD-gated cadence, cross-window resync fix (fresh per-window decoder state + word-level
   seam merge; `TdtFrameNavigation` proved orthogonal). (ADR-015)
5. **Docs, ADRs, final sweep** (this document) — ADR-013/014/015, superseded/amended ADRs,
   CLAUDE.md/README.md rewrite.

OS integration (CapsLock hotkey, `inject_text`, TIS language, clipboard — ADR-002/007/009/011),
the GPUI overlay (ADR-006), and packaging (ADR-008) were completed earlier in the project's
history (Phase 02–05, predating the Parakeet rewrite) and are unaffected by the engine swap
except where amended above.

## Verification

- **Batch regression (CI-able, no mic; shipped):** `cargo run -p test-stt-ffi` — file-based
  `transcribe` on `jfk.wav` asserts the JFK quote. The deterministic gate.
- **Streaming smoke (shipped, ADR-015):** the non-ignored, model-gated
  `stream::session` test drives `run_session` with `jfk.wav` in 100 ms chunks and asserts a
  `PartialTranscript` arrives before the final result containing the quote; `streaming_smoke`
  remains as an `#[ignore]`d live-mic variant.
- **Focus/injection:** with the overlay visible, run a session while a text editor is frontmost;
  confirm text is injected there and the overlay never stole focus (ADR-006 criterion 1).
- **Language:** switch macOS keyboard input source; confirm the session's language argument to
  `engine.start_stream`/`transcribe` matches (ADR-009).
- **Overlay PoC:** manual — transparent, click-through, above menu bar, non-activating (ADR-006 criteria).
- **Mic permission:** deny microphone access in System Settings; confirm
  `DictationEvent::Error { recoverable: false }` fires synchronously from
  `mic_authorization_status()` rather than only surfacing after a failed capture attempt (ADR-012).
- **Model/attribution provenance:** `otool -L target/release/vuho | grep -viE '/usr/lib|/System'`
  finds no non-system dylib (ort statically linked, ADR-014); `Contents/Resources/ATTRIBUTION.txt`
  exists in a packaged bundle (ADR-008).
- **Lints/tests:** `cargo clippy --workspace --all-targets` clean (CONSTITUTION rules; known
  `block 0.1.6` upstream warning aside), `cargo test --workspace` green (plain `cargo test` —
  nextest was planned but is not installed).
