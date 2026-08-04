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
