# TODO

- `crates/vuho-ui/src/wiring.rs`'s `run_provisioning_loop` is ~112 lines, over the
  CONSTITUTION rule 28 40-line guideline. It predates the WP7 code-review pass and needs
  decomposition (e.g. splitting the `crossbeam_channel::select!` arms into named helpers) —
  left untouched by WP7 to keep that pass scoped to the findings it was asked to fix; flagged
  here per WP7 finding F19 rather than silently skipped.
- `crates/vuho-ui/src/panel.rs`'s `PanelRoot::render_idle_status` is ~59 lines, also over the
  CONSTITUTION rule 28 guideline (it was already ~50 lines before WP7 — F11's fraction-lookup
  and F16's padding-chokepoint fixes both had to touch it, adding a few more). Not one of the
  five functions WP7's F19 named for refactor, so left as pre-existing tech debt rather than a
  drive-by refactor outside that finding's scope — needs the same kind of split
  `render_speech_model_section`/`render_mic_row`/`render_permission_row` got in this same pass
  (e.g. extracting the headline/sub-line block and the Downloading-progress-bar block).
- WP8 (final re-review pass, G9) found the rest of the rule-28 ledger incomplete; appending the
  remaining offenders it found rather than silently leaving them off, per the same finding class
  as F19 above. None of these were touched beyond what WP8's other fixes (G1-G8) required — no
  drive-by refactor:
  - `crates/vuho-ui/src/event_loop.rs`'s `apply_events` is ~85 lines (already ~79 before WP8; G4
    added `maybe_show_hud_for_outcome`'s call).
  - `crates/vuho-ui/src/event_loop.rs`'s `spawn_ui_drain` is ~64 lines.
  - `crates/vuho-ui/src/settings_tab.rs`'s `render_hotkey_row` is ~57 lines.
  - `crates/vuho-ui/src/panel.rs`'s `render_tab_button` is ~49 lines.
  - `crates/vuho-ui/src/status_bar.rs`'s `install` is ~49 lines.
  - `crates/vuho-ui/src/overlay.rs`'s `handle_event` is ~48 lines.
- **Esc-to-close regression (ADR-021, single-presentation revision):** `Esc` and `Cmd+,` are gpui
  *window* keybindings, dispatched only to the key window — and the panel now sets
  `setBecomesKeyOnlyIfNeeded: true`, so it is essentially never key, meaning both bindings almost
  never fire in practice. The panel stays closable via its tab-strip "✕" button and the tray-icon
  toggle, so this is not a dead end, just a missing convenience. Possible follow-up: a global
  `CGEventTap`-based `Esc` (parallel to the existing hotkey tap in
  `vuho-os-integration/src/hotkey.rs`) that calls `panel::hide` directly instead of relying on
  gpui's key-window-scoped action dispatch. Not built — out of scope for that revision.
- `scripts/fetch-model.sh`'s `hf_list_files_recursive` swallows the `curl` error (`|| true`), so a
  failed HF tree listing reports only "failed to list <dir> via the HF tree API" and hides the
  real HTTP status (rate limit, 5xx, offline). It also calls the tree API even when every file of
  that directory is already on disk, so an "idempotent" re-fetch fails offline or under rate
  limiting for no reason. Observed once as a transient "failed to list
  ParakeetDecoder.mlmodelc via the HF tree API" that passed on retry. Fix: keep `curl`'s status
  and stderr in the error message, and skip the listing when the lock already names every file
  and each is present at its locked size. Found during ADR-023; left alone to keep that change
  scoped.
- `ParakeetModels::load` and `CanaryModels::load` log a failed warm-up inference and carry on, so
  the failure only surfaces at the first real session. voz's `VozModels::load` now turns a failed
  warm-up into `EngineError::LoadFailed` (fail at load, not at first session); the other two
  backends should follow it so a broken model is refused when it is selected. Found while
  repairing the voz backend; left alone to keep that change scoped.
- **Linux UI and desktop integration (ADR-024, all deferred):**
  - Linux UI: a `gpui` from Zed's git tree with a wlr layer-shell overlay (crates.io `gpui 0.2.2` has none); `vuho-ui` is macOS-only until then.
  - `vuho --toggle`: a CLI that triggers the running instance, bound to a key in Sway (Wayland has no global-hotkey tap).
  - Paste on Linux: `wl-clipboard-rs` plus a `zwp_virtual_keyboard_v1` chord chosen by the focused window's `app_id` (Ctrl+Shift+V in terminals, Ctrl+V elsewhere).
  - Keyboard-layout (language) detection on Linux through Sway IPC.
  - `ksni` tray icon for Linux.
  - Canary and voz on Linux (their assets are CoreML-only).
  - GNOME support (Sway/Wayland is the first target).
  - `evdev` CapsLock trigger on Linux.
  - Silero VAD calling `ort` directly instead of through `voice_activity_detector`, which forces the exact `ort = "=2.0.0-rc.10"` pin.
- Linux real-microphone smoke test: the equivalent of macOS's `#[ignore]`d `streaming_smoke`; only file-driven tests exercise the ONNX streaming session today.
- `crates/vuho-stt-engine/src/parakeet/models.rs` keeps a private `ENCODER_DIM` duplicate of `tdt::ENCODER_DIM`. One-line cleanup (use the shared constant); left because that file only compiles on macOS and the Linux author could not verify the change.
- `CLAUDE.md` says the first unlocked `./scripts/fetch-model.sh <id>` prints `UNLOCKED` and exits non-zero, but the Linux run for the ONNX model exited 0. Pre-existing discrepancy between the docs and the script; verify on macOS which is right, then fix the script or the docs (CLAUDE.md now flags it as unverified).
- ONNX int8 jfk×3 loses punctuation on the third repetition and capitalizes one seam as "Ask" (a window-tail artifact of the int8 decode); content is correct. Not investigated whether it is the model or the seam merge.
