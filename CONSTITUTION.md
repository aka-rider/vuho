# Vuho Engineering Constitution

Rules from code-review findings. Each generalizes a defect class actually found in this repo. All are MUST-level. Terse by design — optimized for automated review.

## Architecture & Data Ownership

1. **One owner per logical resource.** Never mirror one resource across two fields synced by hand (e.g. `stream_handle` + `stream_arc` for one stream). Merge into one field holding one struct.
2. **The producer of a fact is its single source.** Derived data (detected language, confirmed/unconfirmed transcript split) must cross boundaries as data — never fabricated (`language: "en"` downstream of a detector) or re-derived by a consumer (`strip_suffix` reconstruction in the UI).
3. **Match resource lifetime to its real scope.** Expensive resources (ML models, engines) live at app scope and are reused across sessions — never init/load/drop per session.
4. **An event stream has one consumer.** Don't clone every event into two channels because two consumers might exist; give the stream one owner that forwards.
5. **Inject side-effectful dependencies at construction** (engine, text injector). No `#[cfg(test)]` statics or conditional-compilation seams — they make the tested path differ from the shipped path.
6. **Declare only dependencies you use.** An unused `Cargo.toml` edge misstates the architecture.
7. **Mandated stacks:** `objc2` family for CGEvent, TIS, Accessibility. Clipboard: GPUI's API inside GPUI crates; `NSPasteboard` via objc2 in non-GPUI crates. Never `arboard`.

## State Machines & Concurrency

8. **Enumerate every (state, command) pair in an explicit match.** Never route commands through a shared helper whose polarity can invert a transition (the `Stop`-while-recording → `handle_start` class).
9. **The stopper owns the stop signal.** If `stop()` joins a thread, `stop()` must hold and set whatever makes that thread exit. An unreachable stop flag is an immortal thread and a deadlocked join.
10. **Every recv loop handles `Disconnected` explicitly** (exit or propagate). Ignoring `Err` from a closed channel inside `select!` busy-spins.
11. **Emit success events only after the operation succeeded.** `SessionStarted` before engine init lies to the UI.
12. **Poison recovery = log + `into_inner` + continue the normal path.** Never write bespoke cleanup inside poison arms — it's untested code with its own bugs (`Box::from_raw` on an `Arc` pointer is UB).

## FFI Boundaries

13. **Every out-buffer crossing FFI carries its capacity, and the callee bounds all writes.** A count-out parameter without a capacity-in parameter is a heap overflow waiting for input long enough.
14. **Ownership transfer is symmetric:** every `into_raw` has exactly one `from_raw` on every path (success, error, Drop); every callee-allocated pointer is freed on every exit path. Docs claiming "freed/closed" must match the code.
15. **Follow the foreign API's ownership naming rules — and keep the lender alive through every use of what it lent you.** Core Foundation: `Copy`/`Create` = caller owns (`from_raw`); `Get` = borrowed — take an explicit `CFRetained::retain`, never `from_raw` (an over-release). The retain alone isn't the whole fix: don't drop or release the lending object until every use of the borrowed value is finished (the TIS languages-array bug: `retain` was correct, but the owning input source was still dropped before the array was read — an over-release fix left a use-after-free right next to it).
16. **Failure paths leave resources stoppable.** Never deregister a live resource before its shutdown succeeded (registry entry removed → stop fails → orphaned live mic).
17. **Lazy loaders return `Result`** (store `Result` inside `OnceLock`); a panic inside `get_or_init` poisons it permanently.
18. **Validate structural invariants of binary data at the boundary:** computed offsets/lengths fit the buffer; paired data has even length. Error or assert — never silently truncate (`chunks_exact` drops trailing bytes).
19. **Parsers over byte slices return owned data** across crate boundaries.

## Correctness

20. **Own what you create:** store both channel ends and resource handles in the owning struct; release in `Drop`/`stop()`. `let (tx, _rx) = …` makes every send silently fail; `mem::forget(stream)` leaks the mic.
21. **Every parameter and payload flows into the path it implies.** A sender used only on the error path, or `Activity { level }` immediately overwritten with `1.0`, is a dead wire.
22. **Check the reference point of every elapsed-time computation.** `Instant::now().elapsed()` is ~0 — animations driven by it are frozen. Measure from a stored epoch.
23. **Never mix char indices with byte indices** in UTF-8 text — silent corruption on non-ASCII, no panic.
24. **Text cleanup is conservative.** Remove only unambiguous disfluencies; "like", "also", "donc", "thì" are common legitimate words — wholesale removal corrupts dictation. No filler entry may embed whitespace that defeats its own word-boundary check.
25. **Stubs carry a TODO quoting the violated spec line.** A hardcoded value with no TODO reads as a decision.

## Code Quality & Docs

26. **One source of truth per algorithm or format.** WAV parsing, PCM conversion, decay formulas live in exactly one function; a comment saying "matching the logic in X" means you duplicated X — extract instead.
27. **Name semantic numbers** (poll intervals, decay rates, cutoff ratios) as constants.
28. **Functions ≤40 lines and ≤3 responsibilities.** Split FFI flows into prepare / call / read / cleanup.
29. **`#[allow]` at the narrowest scope, on the outer statement.** Never file-wide blankets — `#![allow(dead_code)]` hid an entire unwired overlay show/hide lifecycle.
30. **Don't re-enable lints a parent group covers.** `clippy::all` already includes `correctness`, `style`, `suspicious`, `complexity`, `perf`. (`unwrap_used` lives in `restriction`, not `pedantic`.)
31. **Document behavioral contracts:** post-`stop()` state and restartability, ASCII/Unicode scope of text functions, panic conditions, truncation behavior of bounded buffers.
32. **Never order or pace events with wall-clock sleeps.** A sleep-tuned test hides real races (removing one exposed a genuine feeder/session startup race) and multiplies runtime; inject the clock/cadence as a parameter and synchronize on observable events.
33. **An AppKit call that either delivers synchronous delegate callbacks into gpui or pumps a nested run loop must be issued through `main_queue::defer`, never inline.** `setFrame:display:YES`/`orderFront:`/`orderOut:`/`makeKeyAndOrderFront:` fire delegate callbacks (`windowDidMove:`, `setFrameSize:`) that re-enter gpui through `AsyncApp::update_window`'s non-panicking `try_borrow_mut` — silently dropped if an `App` borrow is already live. `NSAlert::runModal()`/`NSStatusItem`'s `performClick:` pump a nested run loop that can hit `AsyncApp::update`/`update_entity`'s *panicking* `borrow_mut()` from inside it. Deferring lets the current call stack — and any live `App` borrow — unwind first (vuho-ui's `main_queue.rs`).
