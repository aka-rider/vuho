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
