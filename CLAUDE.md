# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Vuho speech-to-text app

WisprFlow clone local-first, fully private.
MacOS Silicon ONLY. ANE. Metal.

100% native integration

1. User activates STT with a global key, always-on-top semi-transparent overlay appears, featuring partial transcription and simple lines waveform
2. STT language always matches OS-native keyboard input method
3. User confirms session end with the hotkey, the app does rule-based text cleanup (filler removal, spacing normalization, newline collapse — see `vuho-postprocess`) and inserts the text into the active window, at the cursor position (so the vuho's own overlay should not grab the focus).
4. The text should be natively send to the focused app, e.g. clipboard + <Cmd>+V keystroke on MacOS

## Where the truth lives

- `ARCHITECTURE.md` — source of truth: 20 ADRs, target architecture, roadmap. Read the relevant ADR before changing anything architectural.
- `CONSTITUTION.md` — 32 MUST-level engineering rules distilled from past code reviews; applied during review.

## Commands

- Build: `cargo build --release -p vuho-ui` — no Swift toolchain required (native CoreML via `objc2-core-ml`).
- Run overlay demo (no mic, no engine): `cargo run -p vuho-ui --features demo`
- Test: `cargo test` (workspace). Single test: `cargo test -p vuho-os-integration map_en_us`. Note: nextest was planned but is NOT installed — use plain `cargo test`.
- Lint: `cargo clippy --workspace --all-targets` (workspace lints enable `clippy::pedantic` at `warn`).
- STT batch regression gate: `cargo run -p test-stt-ffi` must print `PASS` (transcribes `jfk.wav`
  via `ParakeetEngine`; override the audio with `JFK_WAV`). Green as of the Parakeet-TDT rewrite.
- Model provisioning (idempotent, pinned revisions): `./scripts/fetch-model.sh`.
- Package a signed `.app`: `./scripts/package.sh` (delegates to `scripts/bundle-macos.sh`; ad-hoc signature by default, set `SIGN_ID` for a stable identity so TCC grants survive rebuilds).

## Architecture

Cargo workspace of eleven crates, no Swift/FFI — the STT engine is native CoreML via `objc2-core-ml`:

| Crate | Purpose |
|---|---|
| `vuho-domain` | Pure domain types & events (`DictationEvent`, `DictationCommand`, `ModelStatus`); no deps, no platform code |
| `vuho-audio` | cpal capture thread + rubato resample to 16 kHz mono; owns the microphone (ADR-013) |
| `vuho-stt-engine` | `ParakeetEngine`: loads the Parakeet-TDT `CoreML` components (`Preprocessor`/`ParakeetEncoder_15s`/`ParakeetDecoder`/`RNNTJoint`), runs greedy TDT decoding over a sliding 15 s window; Silero VAD via `voice_activity_detector` |
| `vuho-dictation` | Session orchestration: `DictationSession` + `DictationPipeline` (explicit Idle/Recording state machine) |
| `vuho-postprocess` | Rule-based text post-processing: filler removal, spacing normalization, newline collapse; no external deps beyond `vuho-domain` |
| `vuho-os-integration` | macOS via objc2: clipboard, ⌘V injection, TIS keyboard-language detection, CapsLock/chord hotkey (CGEventTap) |
| `vuho-settings` | Serde-only settings persistence: `HotkeySetting` preset + microphone device name, atomic load/save |
| `vuho-model-paths` | Std-only chokepoint crate (ADR-019): embeds `models.manifest.json` and `models.lock.json` at compile time, exposes typed accessors + the single `resolve_model_folder` env-var → bundle → workspace-dev → user-data (ADR-020) resolution chain, plus the shared `atomic_write` helper |
| `vuho-model-fetch` | The only crate permitted to perform network I/O (ADR-020): `availability() -> ModelStatus` (sidecar-and-lock verification, scoped to the user-data candidate only) and `download()` (hf-hub 1.0, Xet-first with HTTPS fallback) |
| `vuho-ui` | GPUI overlay, settings window, and readiness window (permissions + model download, ADR-020); produces the `vuho` binary |
| `test-stt-ffi` | Batch STT regression binary (the `PASS` gate above) |

Data flow: hotkey trigger (CGEventTap thread in `vuho-os-integration/src/hotkey.rs` — CapsLock by default, or a configured chord). CapsLock is **level-triggered**, not edge-triggered (ADR-007): the LED is the source of truth, so a rising `MaskAlphaShift` edge sends `DictationCommand::Start` and a falling edge sends `Stop` (`caps_lock_command`) — a dropped/discarded command desyncs the LED from the session for at most one tap and self-heals rather than permanently inverting the mapping, as an undifferentiated `Toggle` used to. The configured chord still sends `Toggle` (a momentary gesture with no external LED state to track), as does the status-bar menu. Either way, the command travels over a crossbeam channel → `DictationPipeline` starts a session: detects language via the injected `detect_language` policy (TIS in production), reads the configured microphone device from `SettingsStore`, starts streaming capture, forwards `PartialTranscript`/`Activity` events to the UI. `SessionStarted` is emitted **only after** `start_stream` succeeds, so the overlay/waveform never implies listening that isn't happening — `apply_events` therefore also shows the overlay on `Error`, or a failed start would render into a hidden window. On stop: `vuho_postprocess::postprocess(text, language)` (filler removal, spacing normalization, newline collapse) → clipboard + synthesized ⌘V (`inject_text`; Secure Input yields `OsError::SecureInputActive`) → `SessionCompleted`.

> **Streaming is wired**: `ParakeetEngine::start_stream` prechecks mic permission, starts `vuho-audio` capture, and spawns the `"vuho-stt-session"` thread (VAD-gated ≥1 s adaptive partials, `Activity` from capture RMS every ~50 ms, ≥800 ms silence endpoint promotion, end-aligned final window on stop). Decoder state is **fresh per window** — never carried across windows (carry causes blank-lock; continuity comes from the 2 s audio overlap + word-level seam merge, see ADR-015). `stop_stream` on an idle engine returns `EngineError::NoActiveStream`.

Key facts:

- **Audio capture is owned by `vuho-audio`** (ADR-013, superseding the retired WhisperKit-owned-capture design): a dedicated `cpal` thread resamples to 16 kHz mono via `rubato`. The waveform is cosmetic, driven by capture-thread RMS. Device *selection* is by name, resolved via `cpal::default_host().input_devices()`.
- Entry point: `crates/vuho-ui/src/main.rs`. Production wiring (`wire_production`) creates the `DictationSession`, starts the `HotkeyListener` with the persisted preset, installs the status-bar menu (Start/Stop, Settings…, Quit), and registers a `VuhoState` GPUI `Global` (settings store, restartable hotkey listener, command channel, settings-window handle) for the settings window and menu to reach. Quit hotkey is `Cmd+Option+Shift+Q` (the app is `LSUIElement`: no Dock icon, accessory activation policy, non-focus overlay window).
- Settings: `~/.config/vuho/settings.json` (or `$XDG_CONFIG_HOME/vuho/settings.json`), written atomically (temp file + rename); a malformed file logs a warning and falls back to defaults without touching the file. Changing the hotkey in the settings window rebinds the listener **live** (`stop()` + `start()`); changing the microphone applies at the next session start.
- **No FFI, no dylib**: the Parakeet-TDT model components are `.mlmodelc` bundles loaded directly via `objc2-core-ml`'s `MLModel::modelWithContentsOfURL_configuration_error` — no Swift, no `libloading`, no C ABI boundary. `RNNTJoint.mlmodelc`'s real `model.mil` signature takes exactly `encoder_outputs`/`decoder_outputs` (no `encoder_length`, despite early planning notes assuming one); its `logits` output is post-softmax log-probabilities, not raw logits — immaterial for greedy argmax decoding.
- **The engine is app-scoped, loaded once** (CONSTITUTION rule 3): `ParakeetEngine::load(folder)` is the only constructor; there is no `init`/`load_models` step, so an unloaded engine is unrepresentable. `wire_production` warms it on a background thread (status bar shows `Loading model…`) and only then builds the `DictationSession`; presses during warmup are discarded. `ParakeetModels::load` also runs one warmup inference on a zeroed 15 s window to trigger `CoreML`'s ANE plan compilation for `ParakeetEncoder_15s` (the only component loaded with `CpuAndNeuralEngine` — `ParakeetDecoder`/`RNNTJoint` have `RangeDims`-flexible inputs and are CPU-only; loading them with the ANE enabled produces a `CoreML` E5RT validation error at prediction time). **Measured warmup time on this development machine: ~50ms** (`ParakeetModels::load`'s `log::info!` line) — but every measurement taken during this work was against a machine whose CoreML ANE compilation cache was already warm from many prior runs in the same session; a genuine first-ever cold-compile number (which the retired WhisperKit path measured at 136s) was not obtainable without clearing the system-wide CoreML cache, which risks affecting other apps and was judged out of scope. Re-measure on a truly clean machine/cache before trusting a cold number here.
- **Model resolution is a Rust chokepoint, one level deeper than before (ADR-019)** — `vuho_stt_engine::resolve_model_folder()` is a thin wrapper that builds a `vuho_model_paths::ModelSpec` from the embedded `models.manifest.json` and delegates to `vuho_model_paths::resolve_model_folder()`, the **single** copy of the env-var → the enclosing `.app`'s `Contents/Resources/<name>` → the workspace-relative `models/<name>` → `~/Library/Application Support/Vuho/models/<name>` (ADR-020, user-data candidate) fallback chain → `EngineError::ModelFolderMissing`. Vuho's own Rust code never downloads a model at runtime from inside this chokepoint — only `vuho-model-fetch`, gated behind an explicit user action, ever performs network I/O (ADR-020). `ParakeetEngine::load` takes the resolved folder, so an engine without one is unrepresentable.
- Env vars (names themselves live in `models.manifest.json`, not hardcoded in Rust or the scripts): `VUHO_MODEL_FOLDER`, `VUHO_MODEL_NAME` (default `parakeet-tdt-0.6b-v3-coreml`), `JFK_WAV`. There is no "must match" comment to keep in sync anymore — the manifest is the one place these strings are written down; the provisioning/packaging scripts read the same file via `scripts/manifest-lib.sh`.
- **Two distribution shapes, one build script (ADR-020):** the STT model (~474 MB, dominated by `ParakeetEncoder_15s.mlmodelc`'s weights) can either be embedded — provisioned out-of-band via `./scripts/fetch-model.sh` into `models/` (gitignored) and bundled into `Contents/Resources/` by `scripts/bundle-macos.sh` (`VUHO_BUNDLE_MODEL=1`, the DMG build, ≈500 MB, offline from first launch) — or omitted (`VUHO_BUNDLE_MODEL=0`, the Homebrew-cask build, measured ≈40 MB on disk / ≈15 MB as the gzipped release tarball), in which case `vuho-model-fetch` downloads it into `~/Library/Application Support/Vuho/models/` on first run, gated by an explicit Download button in the readiness window and verified against the committed `models.lock.json`.

## The Stack

- Rust; GUI framework gpui 0.2.2 https://gpui.rs/
- Transcription: Parakeet-TDT (`FluidInference/parakeet-tdt-0.6b-v3-coreml`, itself a `CoreML` export of `nvidia/parakeet-tdt-0.6b-v3`) loaded directly via `objc2-core-ml` — a hand-rolled greedy TDT decode loop (`vuho-stt-engine/src/parakeet/tdt.rs`) over Preprocessor → Encoder → (Decoder + Joint) per 15 s sliding window. WhisperKit (Swift, FFI/`libloading`) was retired: out of the project's control and the main source of instability. whisper-rs was rejected before that: no streaming.
- objc2 family (objc2-app-kit, objc2-foundation, objc2-core-graphics, objc2-core-foundation, objc2-core-ml, objc2-av-foundation) for clipboard, CGEvent, TIS input sources, and now CoreML inference (ADR-002: objc2, never arboard)
- cpal + rubato for microphone capture and resampling (`vuho-audio`); `voice_activity_detector` (embedded Silero v5) for VAD
- Post-processing: `vuho-postprocess` — rule-based filler removal, spacing normalization, newline collapse (no external deps beyond `vuho-domain`)
- crossbeam-channel for cross-thread events

## Platform & testing notes

- Apple Silicon macOS 14.0+ only; Metal (via GPUI) required to build. No Swift toolchain required — the STT engine is native CoreML via `objc2-core-ml`.
- Accessibility and Input Monitoring are runtime TCC grants, not entitlements — the only entitlement is `com.apple.security.device.audio-input`. macOS prompts on first launch; reset with `tccutil reset Microphone|Accessibility|InputMonitoring tech.iurii.vuho`.
- Unit tests live inline (`#[cfg(test)] mod tests`). Tests touching macOS APIs (clipboard, TIS) tolerate failure in headless environments; `vuho-dictation` pipeline tests inject a local fake `TranscriptionEngine` via `new_with_engine` (trait-only coupling). `vuho-stt-engine`'s own pure modules (`parakeet::tdt`, `stream::windower`, `stream::merge`, `parakeet::vocab`) unit-test without the model; a handful of tests (`coreml::tests`, `vad::tests`, `tests/batch_multiwindow.rs`) are model-gated and skip cleanly (with an `eprintln`, not a failure) when `models/` is absent.
- Real mic streaming is only exercised via `#[ignore]` tests (`streaming_smoke`); the deterministic gate is the `test-stt-ffi` binary (batch transcription only — `start_stream`/`stop_stream` aren't wired yet, see above). `tests/batch_multiwindow.rs`'s multi-window seam test runs un-ignored (model-gated, skips cleanly without `models/`) and passes locally, verifying no dropped or duplicated content at the window seams.

## Known Issues

- **`block v0.1.6` future-incompatibility warning** — `cargo clippy --workspace --all-targets` reports a `static of uninhabited type` warning from `block 0.1.6` (pulled in by `gpui 0.2.2 → cocoa/metal/core-graphics2 → block`). The crate is unmaintained (latest published version is still 0.1.6). Will become a hard error in a future Rust release. Resolution requires GPUI to migrate away from the old `cocoa` family to `objc2`. No actionable fix on our side. (`proc-macro-error2` now also appears in the same future-incompat report, pulled in transitively by `ort`/`voice_activity_detector`; same story — no actionable fix on our side.)
- **`E5RT` STL-exception lines on stdout, naming `RNNTJoint.mlmodelc`'s `initial_h`/`linear` shape mismatch — reproduces reliably, root-caused, provably harmless.** Investigated (plan #7, WP8): the two `E5RT encountered an STL exception...` lines reproduce on every run that loads `RNNTJoint.mlmodelc`, including `test-stt-ffi` and `cargo test -p vuho-stt-engine`. Localized precisely by isolation: (1) separating stdout/stderr capture shows the lines are on **stdout** (CoreML's native runtime writes directly to the process's stdout fd — not routed through any Rust `log`/`eprintln`/`Result` path, so no Rust code ever "sees" or masks it), and they print strictly *after* the program's own last `println!` (`=== DONE ===`) — i.e. during static/process teardown, not during load, warmup, or `predict()`. (2) A throwaway probe that loaded **only** `RNNTJoint.mlmodelc` (`ComputeUnits::CpuOnly`) and immediately dropped it with **zero** `predict()` calls still reproduced both lines — ruling out any interaction with the other three models or with real inference shapes. (3) The same probe run against **only** `Preprocessor.mlmodelc` (no `RNNTJoint` loaded at all) produced **zero** `E5RT` lines — pinning the cause specifically to `RNNTJoint.mlmodelc` itself. Conclusion: merely *loading* `RNNTJoint.mlmodelc` — regardless of whether it's ever used for a prediction — causes CoreML's E5RT runtime to internally validate/compile the model's declared `RangeDims`-flexible shape bounds (the "expected 1000" in the message matches the `targets`-length-1000 `RangeDims` ceiling documented in `models.rs`'s doc comment), and that internal validation trips a real shape mismatch baked into the `.mlmodelc`'s own `model.mil` graph — a property of the exported model file (`FluidInference/parakeet-tdt-0.6b-v3-coreml`), not of anything this crate's Rust code does with it. Every actual `Result` this crate observes from `RNNTJoint` predictions succeeds, and `test-stt-ffi`'s transcript output is byte-correct — this is CoreML-internal validation noise, not a masked failure. No actionable fix on our side (would require re-exporting the upstream model). Superseded the plan's original "masked production error" concern, which named `initial_h`/`RNNTJoint`/E5RT only in a doc comment with no reproduction evidence — that reproduction now exists, with root cause, above.
