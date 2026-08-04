//! Production wiring: session + command bridge + hotkey + menu bar + drain.
//! Split out of `main.rs` (WP10) — everything here is
//! `#[cfg(not(feature = "demo"))]`, mirroring the module it came from.
//!
//! WP6 (ARCHITECTURE.md ADR-021): `main.rs` now owns creating the settings
//! store, the `StatusModel`/`SettingsTab` entities, and the panel itself
//! (all needed before `wire_production` runs, to decide whether the
//! permissions/relaunch-blocked path short-circuits it entirely) — this
//! module receives them already built and wires the rest: the dictation
//! session, the provisioning state machine, the hotkey listener, and the
//! status-bar item.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use crossbeam_channel::{unbounded, Receiver, Sender};
use gpui::{App, Entity, WindowHandle};
use vuho_domain::{DictationCommand, DictationEvent, ModelStatus};
use vuho_settings::SettingsStore;

use crate::app_state::UiCommand;
use crate::app_status::{HotkeyState, StatusModel};
use crate::event_loop::{spawn_event_drain, spawn_ui_drain};
use crate::panel::PanelRoot;
use crate::settings_tab::SettingsTab;
use crate::{hotkey_presets, permissions, status_bar};

// ── Provisioning state machine (ADR-020) ───────────────────────────────────

/// Command the Settings tab's Download button (or a "Retry" click on a
/// `Failed` row — same command, same transition) sends to the provisioning
/// thread. `main.rs` owns both ends (CONSTITUTION rule 20): the `Sender`
/// half goes straight into `SettingsTab::new`, the `Receiver` half into
/// [`spawn_warmup_and_bridge`]'s thread alongside `cmd_rx`.
pub(crate) enum ProvisionCommand {
    /// Start a download. Also what "Retry" sends — a failed download and a
    /// fresh one are the same transition (`NeedsModel` → `Downloading`).
    Download,
}

/// The provisioning + dictation state machine driven by
/// [`spawn_warmup_and_bridge`]'s thread.
///
/// Every `(Phase, message)` pair this thread can observe is handled by an
/// exhaustive `match` (CONSTITUTION rule 8) spread across
/// [`on_dictation_command`], [`on_provision_command`], and
/// [`on_download_completed`] — no shared helper whose polarity could invert
/// a transition.
///
/// **Single source of the UI's `ModelStatus` (root-cause fix):** the status
/// that used to be broadcast ad hoc from four different call sites (drifting
/// out of sync with `Phase` itself — the four bugs this replaced) is now
/// *carried by the variant itself* — [`phase_status`] is a pure, total
/// mapping from a `Phase` to the `ModelStatus` it implies, and
/// [`run_provisioning_loop`] is the only place that ever calls
/// `ui_tx.send(UiCommand::ModelStatus(..))` (via [`send_phase_status`]),
/// unconditionally, immediately after every single transition. A `Phase`
/// change without its status reaching the UI is therefore not a bug that can
/// be introduced by a future edit forgetting one of four sites — there is
/// only one site, and it can't be skipped without skipping the transition
/// itself.
enum Phase {
    /// No usable model yet, or a download attempt failed. `status` carries
    /// exactly what the Settings tab/status bar show — always
    /// [`ModelStatus::Missing`] or [`ModelStatus::Failed`] — and is the
    /// same value [`on_provision_command`]'s `Download` (also "Retry")
    /// transition re-fetches size from the lock for.
    NeedsModel(ModelStatus),
    /// A download is in flight on a second thread. `status` is
    /// `Downloading`/`Verifying`, straight from that thread's progress
    /// channel, forwarded into a `Phase` by [`run_provisioning_loop`]'s
    /// `progress_rx` arm. `Dictation` commands are still discarded; another
    /// `Download` command is ignored — the Settings tab swaps the
    /// Download button for disabled "In progress…" text as soon as this
    /// variant is entered (see `settings_tab::SettingsTab::render_speech_model_section`), and
    /// because `provision_rx` is drained one message at a time by this
    /// single thread, there is no window where a second click could race
    /// the first transition — the transition itself, not a later render,
    /// is what a click observes.
    Downloading(ModelStatus),
    /// `vuho_model_fetch::availability()` reports the model is `Ready`, but
    /// `ParakeetEngine::load` itself failed (corrupt/incompatible files,
    /// permission error, out of memory, …) — a fact `Missing`/`Failed`
    /// can't express, since the model bytes are actually fine. `message` is
    /// shown the same way a download `Failed` is (Settings tab Failed
    /// row, "Retry" button — see [`phase_status`]), but Retry from this
    /// phase re-attempts the engine load only (`on_provision_command`'s
    /// `EngineFailed` arm) — never a redundant re-download, which
    /// couldn't fix an out-of-band-provisioned model anyway (ADR-020).
    /// Without this variant this state was a dead end: the model row never
    /// showed a way forward, and — the specific bug this variant fixes —
    /// no status ever reached the UI for this branch at all.
    EngineFailed(String),
    /// Model loaded, engine warmed, session ready to take `DictationCommand`s.
    Ready(vuho_dictation::DictationSession),
}

/// The `ModelStatus` a given [`Phase`] implies — pure and total, so a
/// `Phase` value alone fully determines what the UI shows (see [`Phase`]'s
/// doc comment). The only caller is [`send_phase_status`].
fn phase_status(phase: &Phase) -> ModelStatus {
    match phase {
        Phase::NeedsModel(status) | Phase::Downloading(status) => status.clone(),
        Phase::EngineFailed(message) => ModelStatus::Failed {
            message: message.clone(),
        },
        Phase::Ready(_) => ModelStatus::Ready,
    }
}

/// The **sole** producer of `UiCommand::ModelStatus` in the process — every
/// call site in this module that changes `phase` calls this immediately
/// after, with the new value, and nothing else ever sends this command. See
/// [`Phase`]'s doc comment for why that closes off the four-site drift bug
/// class this state machine replaced.
fn send_phase_status(ui_tx: &Sender<UiCommand>, phase: &Phase) {
    let _ = ui_tx.send(UiCommand::ModelStatus(phase_status(phase)));
}

/// What the download thread reports back to the select loop on completion —
/// delivered as a message (CONSTITUTION rule 32: never a `join()`/sleep).
enum DownloadOutcome {
    Finished,
    Failed(String),
}

/// Wire the real pipeline: session + command bridge + hotkey + menu bar +
/// drain, around the panel/settings/status entities `main.rs` already
/// built.
///
/// Kept short (CONSTITUTION rule 28) by delegating channel setup to
/// [`open_dictation_channels`] and hotkey startup to [`start_hotkey`].
///
/// No `provision_tx` parameter: unlike the old `VuhoState`-global design,
/// `main.rs` hands the Download/Retry button's sender directly to
/// [`SettingsTab::new`] when it builds `settings_tab`, so this function has
/// no need to see it — only `provision_rx`, which the provisioning thread
/// owns.
#[allow(
    clippy::too_many_arguments,
    reason = "each parameter is a distinct piece main.rs already built and must hand off \
              exactly once — the panel handle, the settings store, three channel halves, and \
              two shared entities. Bundling them into a struct would just move the same eight \
              fields into a second place, one call site, without reducing what this function \
              actually needs to wire up"
)]
pub(crate) fn wire_production(
    panel: WindowHandle<PanelRoot>,
    settings: &Arc<SettingsStore>,
    ui_tx: Sender<UiCommand>,
    ui_rx: Receiver<UiCommand>,
    provision_rx: Receiver<ProvisionCommand>,
    status: Entity<StatusModel>,
    settings_tab: &Entity<SettingsTab>,
    cx: &mut App,
) {
    let (event_tx, event_rx, cmd_tx, cmd_rx) = open_dictation_channels();
    // Clone before moving `event_tx` into the bridge below, or dropping it
    // at the end of this function kills the pipeline thread.
    let event_tx_bridge = event_tx.clone();

    // The sole `install` call reaching production (the other, `main.rs`'s
    // gate path, always `return`s first); paints the tray itself (F21).
    status_bar::install(
        Some(cmd_tx.clone()),
        ui_tx.clone(),
        &status.read(cx).composite(),
    );

    spawn_warmup_and_bridge(
        cmd_rx,
        provision_rx,
        event_tx,
        event_tx_bridge,
        ui_tx,
        settings.clone(),
    );

    let hotkey = start_hotkey(&cmd_tx, settings, &status, cx);
    settings_tab.update(cx, |tab, _cx| {
        tab.connect_hotkey(Rc::new(RefCell::new(hotkey)), cmd_tx);
    });

    spawn_event_drain(panel, event_rx, status.clone(), cx);
    spawn_ui_drain(panel, ui_rx, status, cx);
}

/// Open the two channel pairs [`wire_production`]'s pipeline runs on — a
/// `DictationEvent` channel (the pipeline's own progress feed) and a
/// `DictationCommand` channel (hotkey/menu → pipeline) — install the TIS
/// keyboard-language watcher, and fire the proactive microphone-permission
/// prompt on the freshly opened event channel. Split out of
/// [`wire_production`] (CONSTITUTION rule 28).
///
/// Returns `(event_tx, event_rx, cmd_tx, cmd_rx)`.
fn open_dictation_channels() -> (
    Sender<DictationEvent>,
    Receiver<DictationEvent>,
    Sender<DictationCommand>,
    Receiver<DictationCommand>,
) {
    // TIS is main-thread-only (uncatchable SIGTRAP off-main): install the
    // keyboard-language watcher here, on the main thread, so the pipeline
    // thread reads its cache instead of ever touching TIS itself.
    vuho_os_integration::install_language_watcher(
        objc2::MainThreadMarker::new()
            .expect("open_dictation_channels runs on the main thread (called from wire_production)"),
    );

    let (event_tx, event_rx) = unbounded::<DictationEvent>();
    let (cmd_tx, cmd_rx) = unbounded::<DictationCommand>();

    // Request microphone permission proactively on the main thread.
    // macOS TCC dialogs only appear reliably from the main run-loop;
    // if we wait until vuho_start_stream (called from a background thread),
    // the dialog may never appear and the stream fails silently.
    request_mic_permission_on_startup(&event_tx);

    (event_tx, event_rx, cmd_tx, cmd_rx)
}

/// Check microphone permission status at app startup.
///
/// `vuho_stt_engine::mic_permission_status()` reads `AVCaptureDevice`'s TCC
/// authorization status via `vuho-audio` (no `CoreML` model, no engine,
/// involved — this check works before/independent of the warmup in
/// [`spawn_warmup_and_bridge`]). On macOS, the system TCC dialog itself is
/// triggered automatically by the first real capture attempt: either this
/// function's own `NotDetermined` branch (which calls
/// `vuho_stt_engine::request_mic_permission`, itself firing
/// `request_mic_access_async` once, fire-and-forget), or, failing that,
/// `ParakeetEngine::start_stream`'s own synchronous precheck the first time
/// a dictation session actually starts. This function only checks and
/// warns/emits; it doesn't block app startup on the user's answer.
///
/// Emits a `DictationEvent::Error { kind: MicPermissionDenied, .. }` on
/// `Denied`/`Restricted`, so the overlay's event drain prompts the user to
/// grant access in System Settings (`event_loop::apply_events`'s
/// `MicPermissionDenied` arm → `permissions::show_microphone_denied`). Under
/// normal startup this branch is unreachable — the ADR-016 preflight gate
/// (`readiness::missing_permissions`) already blocks the app before
/// `wire_production` (and thus this function) ever runs unless Microphone
/// access is `Authorized` — but it closes the TOCTOU window where the grant
/// is revoked in System Settings between the gate's check and this call,
/// which would otherwise leave the user with only the silent log lines this
/// function used to emit and no on-screen signal at all.
///
/// Safe to emit synchronously here despite `wiring.rs`'s general "no modal
/// while the top-level `Application::run` closure holds the app context
/// borrowed" hazard (see [`start_hotkey`]'s doc comment): sending on
/// `event_tx` is not itself modal — the actual `NSAlert::runModal()` call
/// happens later, inside `spawn_event_drain`'s task, which GPUI's
/// `ForegroundExecutor` dispatches to run *after* this closure returns
/// (`ForegroundExecutor::spawn`'s doc: "Enqueues the given Task to run on
/// the main thread at some point in the future" — not synchronously inline).
fn request_mic_permission_on_startup(event_tx: &crossbeam_channel::Sender<DictationEvent>) {
    use vuho_stt_engine::MicAuthStatus;

    match vuho_stt_engine::mic_permission_status() {
        MicAuthStatus::Authorized => log::info!("mic permission: already granted"),
        MicAuthStatus::NotDetermined => {
            // Fire the native TCC dialog now, on the main thread — dialogs
            // only appear reliably from the main run-loop; waiting until a
            // background thread's stream start risks the dialog never
            // appearing at all (see `request_mic_permission`'s own doc).
            let _ = vuho_stt_engine::request_mic_permission();
            log::info!("mic permission: not yet determined — system prompt triggered");
        }
        MicAuthStatus::Denied | MicAuthStatus::Restricted => {
            log::warn!("vuho: microphone access denied");
            log::warn!("vuho: grant access in System Settings → Privacy & Security → Microphone");
            let _ = event_tx.send(DictationEvent::Error {
                message: "Microphone access denied. Grant it in System Settings → Privacy & \
                          Security → Microphone."
                    .to_string(),
                recoverable: true,
                kind: vuho_domain::ErrorKind::MicPermissionDenied,
            });
        }
    }
}

/// Load the engine (once the model is available) off the main thread, then
/// run the provisioning + dictation state machine for the rest of the
/// process's life.
///
/// One thread does all of this because a `DictationSession` cannot exist
/// until an engine is loaded, and an engine cannot load until
/// `vuho_model_fetch::availability()` reports [`ModelStatus::Ready`] — see
/// ADR-020. "Loaded" means `ParakeetEngine::load`, which resolves and loads
/// the four Parakeet-TDT `CoreML` component bundles (~490 MB, no
/// Swift/dylib involved — see `vuho-stt-engine`'s crate doc) and runs one
/// warmup inference to trigger `CoreML`'s ANE plan compilation for the
/// encoder. Measured warm-cache load time on the development machine is
/// ~50 ms, but no genuine cold-ANE-cache number has been obtained (see
/// `CLAUDE.md`'s caveat) — this thread exists precisely so that whatever
/// the real number turns out to be on a given machine, it never blocks the
/// command loop. Doing this work inside the pipeline's `handle_start` — as
/// an earlier (`WhisperKit`-era) version did — stalled the first hotkey
/// press *and* blocked the command loop behind it, so the overlay opened
/// and never closed; the same risk applies to any future load path here,
/// warm or cold.
///
/// Once loaded, the engine is handed to a `DictationSession`, whose
/// pipeline drives this same engine's `start_stream`/`stop_stream`
/// (`vuho-stt-engine`'s `"vuho-stt-session"` thread) for every dictation
/// session for the rest of the process's life — the warmup here happens
/// exactly once, not per session (CONSTITUTION rule 3).
///
/// The thread returns *only* when both `cmd_rx` and `provision_rx`
/// disconnect (CONSTITUTION rule 10) — never on a model or download error.
/// The previous version of this function `return`ed on a warmup failure,
/// dropping `cmd_rx` and turning every later hotkey press into a silent
/// no-op forever; that bug cannot recur here because a model/engine failure
/// only ever moves [`Phase`], never ends the loop.
fn spawn_warmup_and_bridge(
    cmd_rx: Receiver<DictationCommand>,
    provision_rx: Receiver<ProvisionCommand>,
    event_tx: Sender<DictationEvent>,
    event_tx_bridge: Sender<DictationEvent>,
    ui_tx: Sender<UiCommand>,
    settings: Arc<SettingsStore>,
) {
    std::thread::spawn(move || {
        // Keep `event_tx_bridge` alive for the process lifetime so the pipeline
        // thread's event sender never gets dropped.
        let _event_tx_bridge = event_tx_bridge;
        let initial = initial_phase(&event_tx, &ui_tx, &settings);
        run_provisioning_loop(
            &cmd_rx,
            &provision_rx,
            &event_tx,
            &ui_tx,
            &settings,
            initial,
            spawn_download_thread,
        );
    });
}

/// Drop any [`DictationCommand`]s queued while a blocking model/engine load
/// ran on this thread — the initial startup check, a post-download engine
/// load, or an [`Phase::EngineFailed`] retry — matching the documented
/// "presses during warmup are discarded" behavior. Without this (A1), a
/// `CapsLock` press landing mid-load is still sitting in `cmd_rx` once
/// `Phase::Ready` is reached, and is delivered immediately on the next
/// iteration — a session nobody consciously started.
fn drain_stale_dictation_commands(cmd_rx: &Receiver<DictationCommand>) {
    for cmd in cmd_rx.try_iter() {
        log::info!("provisioning: discarding {cmd:?} queued during a blocking load");
    }
}

/// The provisioning + dictation state machine's event loop.
///
/// `initial` is the starting [`Phase`], already resolved by the caller
/// (production: [`initial_phase`], which touches `vuho_model_fetch` and
/// possibly loads the engine; tests: any `Phase` value, injected directly —
/// this is what makes the loop itself testable without network or a real
/// engine, per this module's test module). `spawn_download` is the
/// side-effecting "start a download thread" step, also injected
/// (CONSTITUTION rule 5): production always passes [`spawn_download_thread`];
/// tests pass a fake under their own control.
///
/// Sends `initial`'s status once (see [`send_phase_status`]) before doing
/// anything else, then drops anything pressed during whatever synchronous
/// work produced it, then services `cmd_rx`/`provision_rx`/download-progress/
/// download-completion messages with `crossbeam_channel::select!` for the
/// rest of the process's life. `download_done_rx`/`progress_rx` start as
/// [`crossbeam_channel::never`] — a receiver that never becomes ready — and
/// are swapped for real ones only while [`Phase::Downloading`], so `select!`
/// never has to special-case "no download in flight" as a `Disconnected`
/// error.
fn run_provisioning_loop(
    cmd_rx: &Receiver<DictationCommand>,
    provision_rx: &Receiver<ProvisionCommand>,
    event_tx: &Sender<DictationEvent>,
    ui_tx: &Sender<UiCommand>,
    settings: &Arc<SettingsStore>,
    initial: Phase,
    mut spawn_download: impl FnMut() -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>),
) {
    let mut phase = initial;
    send_phase_status(ui_tx, &phase);
    drain_stale_dictation_commands(cmd_rx);

    let mut download_done_rx: Receiver<DownloadOutcome> = crossbeam_channel::never();
    let mut progress_rx: Receiver<ModelStatus> = crossbeam_channel::never();
    loop {
        crossbeam_channel::select! {
            recv(cmd_rx) -> msg => {
                let Ok(cmd) = msg else {
                    log::info!("provisioning: dictation channel disconnected — stopping");
                    return;
                };
                phase = on_dictation_command(phase, cmd);
            },
            recv(provision_rx) -> msg => match msg {
                Ok(ProvisionCommand::Download) => {
                    let (next, outcome) = on_provision_command(
                        phase,
                        ui_tx,
                        event_tx,
                        settings,
                        &mut spawn_download,
                    );
                    phase = next;
                    match outcome {
                        ProvisionOutcome::DownloadStarted(done_rx, prog_rx) => {
                            download_done_rx = done_rx;
                            progress_rx = prog_rx;
                        }
                        // `Phase::EngineFailed`'s retry reloads the engine
                        // in place on this thread, blocking it — drain any
                        // presses that landed during that (A1), same as the
                        // startup path above.
                        ProvisionOutcome::RetriedBlocking => {
                            drain_stale_dictation_commands(cmd_rx);
                        }
                        ProvisionOutcome::Ignored => {}
                    }
                    send_phase_status(ui_tx, &phase);
                }
                Err(_) => {
                    log::info!("provisioning: provision channel disconnected — stopping");
                    return;
                }
            },
            recv(progress_rx) -> msg => {
                match msg {
                    Ok(status) => {
                        phase = Phase::Downloading(status);
                        send_phase_status(ui_tx, &phase);
                    }
                    Err(_) => {
                        // The download thread finished (or died) and
                        // dropped its progress sender — nothing more to
                        // forward on this channel. Fall back to `never()`
                        // so `select!` doesn't spin hot on a now-permanently-
                        // ready `Disconnected` receiver (CONSTITUTION rule
                        // 10); the actual completion still arrives on
                        // `download_done_rx` below.
                        progress_rx = crossbeam_channel::never();
                    }
                }
            },
            recv(download_done_rx) -> msg => {
                // Reset BOTH receivers to `never()` before anything else —
                // and `progress_rx` too, not only `download_done_rx`
                // (CONSTITUTION rule 10's hot-spin hazard applies to
                // either), which is also what closes the stale-progress
                // race (B3): once a completion is observed, any progress
                // message still buffered and unread is simply dropped
                // instead of being applied to `phase` on a later
                // iteration — the worst case is one stale render already
                // in flight, immediately followed by the terminal one this
                // arm sends, never a permanently wedged "Downloading".
                download_done_rx = crossbeam_channel::never();
                progress_rx = crossbeam_channel::never();
                // An `Err` here means the sender was dropped without
                // sending: the download thread died (panicked) mid-flight.
                // Nothing else will ever report that, so synthesize the
                // failure — otherwise the UI sits on "Downloading" forever
                // with no download running and no way to retry.
                let outcome = msg.unwrap_or_else(|_| {
                    log::error!("provisioning: download thread died without reporting");
                    DownloadOutcome::Failed(
                        "the download stopped unexpectedly — please try again".to_owned(),
                    )
                });
                phase = on_download_completed(phase, outcome, event_tx, ui_tx, settings);
                send_phase_status(ui_tx, &phase);
                // Only the `Finished` path runs a blocking engine load
                // (`load_after_download`) on this thread — a `Failed`
                // outcome does no blocking work, so draining here
                // unconditionally would discard a legitimate press that
                // simply happened to race with an unrelated download
                // failure. Restrict the drain to when it actually matters.
                if matches!(phase, Phase::Ready(_)) {
                    drain_stale_dictation_commands(cmd_rx);
                }
            },
        }
    }
}

/// The starting [`Phase`], decided by `vuho_model_fetch::availability()`
/// (ADR-020's chokepoint). `Ready` loads the engine exactly as before this
/// state machine existed — no behavior change for anyone who already has a
/// model; a load failure there now becomes [`Phase::EngineFailed`] rather
/// than the dead-end [`Phase::NeedsModel`] it used to (B4) — the caller
/// ([`run_provisioning_loop`]) sends this phase's status unconditionally, so
/// there is no longer a branch here that can silently withhold it.
fn initial_phase(
    event_tx: &Sender<DictationEvent>,
    ui_tx: &Sender<UiCommand>,
    settings: &Arc<SettingsStore>,
) -> Phase {
    match vuho_model_fetch::availability() {
        ModelStatus::Ready => match load_engine_and_session(event_tx, ui_tx, settings) {
            Some(session) => Phase::Ready(session),
            None => Phase::EngineFailed(
                "the model is present but the engine failed to load at startup — see the log \
                 for details"
                    .to_owned(),
            ),
        },
        other => Phase::NeedsModel(other),
    }
}

/// `(Phase, DictationCommand)`: discard while the model isn't ready yet,
/// forward to the session once it is. One of the three exhaustive match
/// sites CONSTITUTION rule 8 requires for this state machine.
fn on_dictation_command(phase: Phase, cmd: DictationCommand) -> Phase {
    match phase {
        p @ (Phase::NeedsModel(_) | Phase::Downloading(_) | Phase::EngineFailed(_)) => {
            log::info!("provisioning: discarding {cmd:?} — model not ready yet");
            p
        }
        Phase::Ready(session) => {
            log::debug!("bridge: received {cmd:?} from channel");
            if let Err(e) = session.command(cmd) {
                log::error!("bridge: failed to send to pipeline: {e}");
            }
            Phase::Ready(session)
        }
    }
}

/// What [`on_provision_command`] did, for [`run_provisioning_loop`] to act
/// on — explicit rather than an `Option`, so "no download thread started"
/// (an ignored click) and "a blocking engine reload just ran on this
/// thread" (needs [`drain_stale_dictation_commands`], A1) can't be confused
/// with each other.
enum ProvisionOutcome {
    /// [`Phase::NeedsModel`] → [`Phase::Downloading`]: the two receivers
    /// [`run_provisioning_loop`] should install.
    DownloadStarted(Receiver<DownloadOutcome>, Receiver<ModelStatus>),
    /// [`Phase::EngineFailed`]'s retry ran a blocking `ParakeetEngine::load`
    /// on this thread, in place — win or lose, `cmd_rx` needs draining.
    RetriedBlocking,
    /// The command was ignored: already [`Phase::Downloading`], or
    /// [`Phase::Ready`] (unreachable in the shipped UI in practice — once
    /// model and engine are both `Ready`, `settings_tab::should_show_speech_model_section`
    /// hides the Speech Model card, and its Download/Retry button along
    /// with it, so there is nothing left to click that could send this —
    /// but handled rather than left an unspecified match arm). F23: this
    /// does *not* close any window — there is no separate readiness window
    /// left to self-close (ADR-021); the panel (`crate::panel`) simply stays
    /// open on whichever tab the user left it on.
    Ignored,
}

/// `(Phase, ProvisionCommand::Download)`: start a download from
/// [`Phase::NeedsModel`] (returning the new [`Phase::Downloading`], whose
/// status is set synchronously to `Downloading { received_bytes: 0, .. }`
/// rather than waiting for the download thread's first progress tick — A3:
/// the Settings tab's button reflects the click immediately, not one
/// network round-trip later); retry a blocking engine load from
/// [`Phase::EngineFailed`] (B4 — no redundant re-download, which can't fix
/// an out-of-band-provisioned model anyway); ignore it while already
/// [`Phase::Downloading`] or [`Phase::Ready`].
///
/// `spawn` is the side-effecting "start a download thread" step, injected
/// rather than hardcoded (CONSTITUTION rule 5) so this transition can be
/// tested without touching the network — production always passes
/// [`spawn_download_thread`]; tests pass a fake that returns receivers under
/// their own control.
fn on_provision_command(
    phase: Phase,
    ui_tx: &Sender<UiCommand>,
    event_tx: &Sender<DictationEvent>,
    settings: &Arc<SettingsStore>,
    spawn: &mut impl FnMut() -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>),
) -> (Phase, ProvisionOutcome) {
    match phase {
        Phase::NeedsModel(_) => {
            log::info!("provisioning: starting model download");
            let total_bytes = vuho_model_paths::lock().stt.total_bytes;
            let (done_rx, progress_rx) = spawn();
            (
                Phase::Downloading(ModelStatus::Downloading {
                    received_bytes: 0,
                    total_bytes,
                }),
                ProvisionOutcome::DownloadStarted(done_rx, progress_rx),
            )
        }
        Phase::EngineFailed(_) => {
            log::info!("provisioning: retrying engine load");
            let next = match load_engine_and_session(event_tx, ui_tx, settings) {
                Some(session) => Phase::Ready(session),
                None => Phase::EngineFailed(
                    "the model is present but the engine still failed to load — see the log \
                     for details"
                        .to_owned(),
                ),
            };
            (next, ProvisionOutcome::RetriedBlocking)
        }
        downloading @ Phase::Downloading(_) => {
            log::info!("provisioning: download already in progress — ignoring");
            (downloading, ProvisionOutcome::Ignored)
        }
        ready @ Phase::Ready(_) => {
            log::warn!("provisioning: Download command while already Ready — ignoring");
            (ready, ProvisionOutcome::Ignored)
        }
    }
}

/// Download completion, delivered as a message (never a `join()`/sleep —
/// CONSTITUTION rule 32). Only meaningful from [`Phase::Downloading`]; any
/// other phase observing this message would mean a stale completion from an
/// earlier download landed after a state change, so it's passed through
/// unchanged rather than acted on.
fn on_download_completed(
    phase: Phase,
    outcome: DownloadOutcome,
    event_tx: &Sender<DictationEvent>,
    ui_tx: &Sender<UiCommand>,
    settings: &Arc<SettingsStore>,
) -> Phase {
    match (phase, outcome) {
        (Phase::Downloading(_), DownloadOutcome::Finished) => {
            load_after_download(event_tx, ui_tx, settings)
        }
        (Phase::Downloading(_), DownloadOutcome::Failed(message)) => {
            log::error!("provisioning: download failed: {message}");
            Phase::NeedsModel(ModelStatus::Failed { message })
        }
        (other, _) => other,
    }
}

/// After a successful download, re-derive readiness from
/// `availability()` rather than trusting the download thread's own success
/// signal — `ParakeetEngine::load` must never run until the one chokepoint
/// that decides trustworthiness says `Ready` (ADR-020's enforcement clause).
fn load_after_download(
    event_tx: &Sender<DictationEvent>,
    ui_tx: &Sender<UiCommand>,
    settings: &Arc<SettingsStore>,
) -> Phase {
    match vuho_model_fetch::availability() {
        ModelStatus::Ready => match load_engine_and_session(event_tx, ui_tx, settings) {
            Some(session) => Phase::Ready(session),
            None => Phase::EngineFailed(
                "the model finished downloading but the engine failed to load — see the log \
                 for details"
                    .to_owned(),
            ),
        },
        other => {
            log::error!("provisioning: download finished but availability() reports {other:?}");
            Phase::NeedsModel(other)
        }
    }
}

/// Resolve and load the engine, then build the `DictationSession` around it.
/// Shared by [`initial_phase`] (model already present at startup),
/// [`load_after_download`] (model just finished downloading), and
/// [`on_provision_command`]'s `Phase::EngineFailed` retry (model present,
/// engine previously failed to load) — one load path, not three
/// (CONSTITUTION rule 26).
fn load_engine_and_session(
    event_tx: &Sender<DictationEvent>,
    ui_tx: &Sender<UiCommand>,
    settings: &Arc<SettingsStore>,
) -> Option<vuho_dictation::DictationSession> {
    use vuho_stt_engine::{resolve_model_folder, ParakeetEngine};

    log::info!("warmup: loading engine");
    let started = std::time::Instant::now();
    let engine = match resolve_model_folder().and_then(ParakeetEngine::load) {
        Ok(engine) => engine,
        Err(e) => {
            log::error!("warmup: engine unavailable: {e}");
            let _ = ui_tx.send(UiCommand::EngineReady(Err(e.to_string())));
            // Surface it on the overlay too — the menu bar alone is easy to miss.
            let _ = event_tx.send(DictationEvent::Error {
                message: format!("Engine unavailable: {e}"),
                recoverable: false,
                kind: vuho_domain::ErrorKind::Other,
            });
            return None;
        }
    };
    log::info!("warmup: engine ready in {:.1?}", started.elapsed());

    // Settings are injected at construction (CONSTITUTION rule 5) — the
    // pipeline reads the configured microphone device on every session
    // start. The injector is the real ⌘V/clipboard delivery,
    // constructor-injected (CONSTITUTION rule 5) so tests can substitute a
    // fake through the same door.
    let injector: vuho_dictation::Injector = Arc::new(vuho_os_integration::inject_text);
    let session = vuho_dictation::DictationSession::new(
        event_tx.clone(),
        Box::new(engine),
        settings.clone(),
        injector,
    );
    let _ = ui_tx.send(UiCommand::EngineReady(Ok(())));
    Some(session)
}

/// Spawn the thread that performs the blocking download itself, reporting
/// both progress and completion **inward** to [`run_provisioning_loop`] as
/// messages rather than via `join()`/sleep (CONSTITUTION rule 32) — its
/// `select!` polls both returned receivers like any other channel, so the
/// loop stays responsive to `cmd_rx`/`provision_rx` for the whole download.
///
/// Deliberately does **not** touch `ui_tx` — see [`Phase`]'s doc comment.
/// This thread, and the loop it reports to, are the two sides of "the
/// download thread... report[s] inward to the loop; the loop owns the
/// outward transition": before this fix, this function (and a second
/// forwarding thread it used to spawn) sent `UiCommand::ModelStatus`
/// directly, which is exactly what let a stale `Downloading` tick from one
/// sender land after a `Failed` from the other (B3) — two independent
/// senders into the same channel have no ordering guarantee across each
/// other. Now there is only one sender of that command in the whole
/// process (`run_provisioning_loop`'s [`send_phase_status`]), so that race
/// cannot recur structurally, not just by convention.
fn spawn_download_thread() -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>) {
    let (done_tx, done_rx) = crossbeam_channel::bounded(1);
    let (progress_tx, progress_rx) = crossbeam_channel::unbounded::<ModelStatus>();
    std::thread::spawn(move || {
        let outcome = match vuho_model_fetch::download(&progress_tx) {
            Ok(_path) => DownloadOutcome::Finished,
            Err(e) => {
                log::error!("provisioning: download failed: {e}");
                DownloadOutcome::Failed(e.to_string())
            }
        };
        drop(progress_tx);
        let _ = done_tx.send(outcome);
    });
    (done_rx, progress_rx)
}

/// Start the global hotkey with the persisted preset (`CapsLock` by default).
///
/// On failure (Accessibility not granted), prompts for the grant; the menu
/// bar still works and the hotkey binds after a relaunch (or after the user
/// grants access and re-selects a preset in the panel's Settings tab).
///
/// The prompt is deferred via `cx.spawn` rather than called inline: this
/// whole function runs synchronously inside GPUI's top-level `Application::run`
/// closure, which holds the app context borrowed for its entire duration.
/// `prompt_accessibility`'s `NSAlert::runModal()` pumps a nested run loop, and
/// the overlay's animation timer (already ticking) would try to re-borrow the
/// app context from within that nested loop and hit an already-borrowed panic.
/// Deferring lets `Application::run`'s closure return and release its borrow
/// first, so the nested modal loop runs with no outer borrow to conflict with.
fn start_hotkey(
    cmd_tx: &crossbeam_channel::Sender<vuho_domain::DictationCommand>,
    settings: &vuho_settings::SettingsStore,
    status: &gpui::Entity<StatusModel>,
    cx: &mut App,
) -> vuho_os_integration::HotkeyListener {
    let mut hotkey = vuho_os_integration::HotkeyListener::new();
    let preset = settings.get().hotkey;
    let config = hotkey_presets::to_hotkey_config(preset);
    log::info!("hotkey: starting with config {config:?}");
    let new_hotkey_state = if hotkey.start(cmd_tx, config).is_err() {
        log::warn!("hotkey: start failed — Accessibility not granted");
        cx.spawn(move |_cx: &mut gpui::AsyncApp| async move {
            permissions::prompt_accessibility();
        })
        .detach();
        HotkeyState::Failed(preset)
    } else {
        HotkeyState::Active(preset)
    };
    status.update(cx, |model, cx| {
        model.hotkey = new_hotkey_state;
        cx.notify();
    });
    hotkey
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test fixtures ───────────────────────────────────────────────────

    fn dummy_event_tx() -> Sender<DictationEvent> {
        crossbeam_channel::unbounded().0
    }

    fn dummy_settings() -> Arc<SettingsStore> {
        Arc::new(SettingsStore::new_temp("wiring-provisioning-test"))
    }

    /// `on_provision_command`'s `NeedsModel` arm re-fetches `total_bytes`
    /// from the repo-pinned lock (not from whatever total the `Phase` it's
    /// leaving happened to carry) — this is what a `Downloading{0, ..}`
    /// transition's `total_bytes` actually equals in every test below.
    fn lock_total_bytes() -> u64 {
        vuho_model_paths::lock().stt.total_bytes
    }

    /// A `spawn_download` fake that is never actually called in a given
    /// test (used where the test only cares about a phase that ignores or
    /// never reaches `ProvisionCommand::Download`).
    fn unreachable_spawn() -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>) {
        panic!("spawn_download must not be called in this test");
    }

    /// A `TranscriptionEngine` that satisfies `DictationSession::new`'s
    /// bound without touching CoreML/the microphone — these tests exercise
    /// the provisioning state machine's status broadcasting, never a real
    /// dictation session, so every method here is unreachable in practice.
    /// Mirrors `vuho-dictation`'s own `StubEngine` (`lib.rs` tests).
    struct FakeEngine;
    impl vuho_stt_engine::TranscriptionEngine for FakeEngine {
        fn transcribe(
            &self,
            _samples: &[f32],
            _language: Option<&str>,
        ) -> Result<vuho_domain::TranscriptionResult, vuho_stt_engine::EngineError> {
            Ok(vuho_domain::TranscriptionResult {
                segments: Vec::new(),
                full_text: String::new(),
                language: "en".to_owned(),
            })
        }
        fn unload(&self) {}
        fn start_stream(
            &self,
            _language: Option<&str>,
            _input_device: Option<&str>,
        ) -> Result<Receiver<DictationEvent>, vuho_stt_engine::EngineError> {
            Err(vuho_stt_engine::EngineError::NoActiveStream)
        }
        fn stop_stream(
            &self,
        ) -> Result<vuho_domain::TranscriptionResult, vuho_stt_engine::EngineError> {
            Err(vuho_stt_engine::EngineError::NoActiveStream)
        }
    }

    /// A `Phase::Ready` built from [`FakeEngine`] — for tests of the
    /// `Ready` arm that don't need a real model or a real dictation
    /// session, only a `Phase` value of the right shape.
    fn fake_ready_phase() -> Phase {
        let injector: vuho_dictation::Injector = Arc::new(|_: &str| Ok(()));
        Phase::Ready(vuho_dictation::DictationSession::new(
            dummy_event_tx(),
            Box::new(FakeEngine),
            dummy_settings(),
            injector,
        ))
    }

    /// Upper bound on how long a test waits for a status the loop is
    /// expected to emit.
    ///
    /// Bounded, not a bare `recv()`: the defects these tests guard are
    /// *omitted* sends, so a regression means the status never arrives. A
    /// blocking `recv()` turns that into a hung test that burns the whole
    /// CI job timeout with no diagnostic; `recv_timeout` turns it into a
    /// named assertion failure in a second. This is a failure-mode bound,
    /// not event pacing — CONSTITUTION rule 32 forbids ordering work with
    /// sleeps, and nothing here is ordered by it: on healthy code the
    /// status is already queued and the wait returns immediately.
    const STATUS_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

    /// Block for the next `ModelStatus`, failing the test (rather than
    /// hanging) if the loop never emits one.
    fn next_status(ui_rx: &crossbeam_channel::Receiver<UiCommand>) -> Option<ModelStatus> {
        loop {
            match ui_rx.recv_timeout(STATUS_WAIT) {
                Ok(cmd) => {
                    if let Some(status) = model_status(cmd) {
                        return Some(status);
                    }
                }
                // Distinguish the two: a timeout means the loop is alive but
                // withheld a send (the defect class these tests guard); a
                // disconnect means the loop thread itself exited or panicked.
                // Reporting them identically would send the next reader
                // hunting for a missing send that never was the problem.
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => panic!(
                    "timed out after {STATUS_WAIT:?} waiting for a ModelStatus — \
                     the provisioning loop never sent one for this transition"
                ),
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    panic!(
                        "the provisioning loop dropped its ui_tx — the thread exited or panicked"
                    )
                }
            }
        }
    }

    /// Pull the `ModelStatus` out of a `UiCommand`, discarding anything
    /// else (`EngineReady`, `OpenPanel`, …) — what these tests care about
    /// is exactly the one command [`send_phase_status`] produces.
    fn model_status(cmd: UiCommand) -> Option<ModelStatus> {
        match cmd {
            UiCommand::ModelStatus(status) => Some(status),
            _ => None,
        }
    }

    /// Run [`run_provisioning_loop`] on a background thread (it blocks),
    /// returning the two command senders the test drives it with and the
    /// `UiCommand` receiver it broadcasts on. The loop only returns once
    /// **both** returned senders have been dropped (CONSTITUTION rule 10 —
    /// see the loop's own doc comment) — callers must drop them and
    /// `join()` the handle to avoid leaking the thread past the test.
    fn spawn_loop<F>(
        initial: Phase,
        spawn_download: F,
    ) -> (
        Sender<DictationCommand>,
        Sender<ProvisionCommand>,
        Receiver<UiCommand>,
        std::thread::JoinHandle<()>,
    )
    where
        F: FnMut() -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>) + Send + 'static,
    {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<DictationCommand>();
        let (provision_tx, provision_rx) = crossbeam_channel::unbounded::<ProvisionCommand>();
        let (ui_tx, ui_rx) = crossbeam_channel::unbounded::<UiCommand>();
        let handle = std::thread::spawn(move || {
            let event_tx = dummy_event_tx();
            let settings = dummy_settings();
            run_provisioning_loop(
                &cmd_rx,
                &provision_rx,
                &event_tx,
                &ui_tx,
                &settings,
                initial,
                spawn_download,
            );
        });
        (cmd_tx, provision_tx, ui_rx, handle)
    }

    // ── phase_status — pure mapping, one test per variant ──────────────

    #[test]
    fn phase_status_covers_every_variant() {
        assert_eq!(
            phase_status(&Phase::NeedsModel(ModelStatus::Missing { total_bytes: 7 })),
            ModelStatus::Missing { total_bytes: 7 }
        );
        assert_eq!(
            phase_status(&Phase::Downloading(ModelStatus::Verifying)),
            ModelStatus::Verifying
        );
        assert_eq!(
            phase_status(&Phase::EngineFailed("boom".to_owned())),
            ModelStatus::Failed {
                message: "boom".to_owned()
            }
        );
        assert_eq!(phase_status(&fake_ready_phase()), ModelStatus::Ready);
    }

    // ── B1: nothing ever produced ModelStatus::Ready ────────────────────

    /// Direct regression for B1: `initial_phase`/`load_after_download` used
    /// to match `ModelStatus::Ready` in a branch that sent nothing at all —
    /// `Ready` was consumed, never forwarded, so the Settings tab never
    /// closed after a successful provision. Here `run_provisioning_loop`
    /// is handed a `Phase::Ready` directly (bypassing the real model/engine
    /// entirely — this loop's status broadcasting is what's under test, not
    /// engine loading) and must broadcast `ModelStatus::Ready` for it,
    /// unconditionally, before doing anything else.
    #[test]
    fn entering_ready_phase_broadcasts_model_status_ready() {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<DictationCommand>();
        let (provision_tx, provision_rx) = crossbeam_channel::unbounded::<ProvisionCommand>();
        let (ui_tx, ui_rx) = crossbeam_channel::unbounded::<UiCommand>();
        // Dropped up front: whichever of `cmd_rx`/`provision_rx` the very
        // first `select!` call observes as disconnected, the loop returns
        // immediately either way (both arms `return` on `Err`) — so this
        // runs synchronously, no thread needed, and is not racy.
        drop(cmd_tx);
        drop(provision_tx);
        let event_tx = dummy_event_tx();
        let settings = dummy_settings();

        run_provisioning_loop(
            &cmd_rx,
            &provision_rx,
            &event_tx,
            &ui_tx,
            &settings,
            fake_ready_phase(),
            unreachable_spawn,
        );

        let statuses: Vec<ModelStatus> = ui_rx.try_iter().filter_map(model_status).collect();
        assert_eq!(
            statuses,
            vec![ModelStatus::Ready],
            "B1: entering Phase::Ready must broadcast ModelStatus::Ready — \
             the bug this regresses sent nothing for the Ready arm at all"
        );
    }

    // ── B4: engine-load failure on an available model was a silent sink ─

    /// Direct regression for B4: `initial_phase`'s `Ready`-availability/
    /// engine-load-failure branch used to return `Phase::NeedsModel` with
    /// no status sent at all, so `LAST_MODEL_STATUS` stayed `None` forever
    /// and "Setup…" opened no window — a genuine dead end recoverable only
    /// by deleting the model directory by hand. `Phase::EngineFailed`
    /// replaces that dead end, and — like every other `Phase` — its status
    /// reaches the UI unconditionally, the moment the loop starts.
    #[test]
    fn entering_engine_failed_phase_broadcasts_a_recoverable_failed_status() {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<DictationCommand>();
        let (provision_tx, provision_rx) = crossbeam_channel::unbounded::<ProvisionCommand>();
        let (ui_tx, ui_rx) = crossbeam_channel::unbounded::<UiCommand>();
        drop(cmd_tx);
        drop(provision_tx);
        let event_tx = dummy_event_tx();
        let settings = dummy_settings();

        run_provisioning_loop(
            &cmd_rx,
            &provision_rx,
            &event_tx,
            &ui_tx,
            &settings,
            Phase::EngineFailed("the engine blew up".to_owned()),
            unreachable_spawn,
        );

        let statuses: Vec<ModelStatus> = ui_rx.try_iter().filter_map(model_status).collect();
        assert_eq!(
            statuses,
            vec![ModelStatus::Failed {
                message: "the engine blew up".to_owned()
            }],
            "B4: a Ready-but-engine-failed phase must surface a Failed \
             status (Settings tab Failed row + Retry) rather than \
             leaving the UI with nothing sent for it at all"
        );
    }

    // ── B2: the download-thread-died path updated Phase but never the UI ─

    /// Direct regression for B2: `on_download_completed`'s
    /// `(Downloading, Failed)` arm used to `log::error!` and return
    /// `NeedsModel` with **no** `ui_tx.send` — the UI only ever saw
    /// `Failed` in practice because `spawn_download_thread` sent it
    /// *itself* on its own error path, which a *panicking* thread never
    /// reaches. This drives the loop through exactly that panicking-thread
    /// path: `spawn` hands back an already-disconnected `DownloadOutcome`
    /// receiver (the sender is dropped before returning), so
    /// `run_provisioning_loop` must synthesize the failure itself — and,
    /// under the fix, broadcast it.
    #[test]
    fn a_download_thread_that_dies_without_reporting_ends_in_a_broadcast_failed_status() {
        let (cmd_tx, provision_tx, ui_rx, handle) = spawn_loop(
            Phase::NeedsModel(ModelStatus::Missing { total_bytes: 100 }),
            || {
                let (done_tx, done_rx) = crossbeam_channel::bounded::<DownloadOutcome>(1);
                drop(done_tx); // the "thread" died before reporting
                (done_rx, crossbeam_channel::never())
            },
        );

        // The initial `Missing` status, sent before the loop services
        // `provision_rx` at all.
        assert_eq!(
            next_status(&ui_rx),
            Some(ModelStatus::Missing { total_bytes: 100 })
        );

        provision_tx.send(ProvisionCommand::Download).unwrap();

        // The synchronous `Downloading{0,total}` transition (A3), then the
        // synthesized `Failed` from the dead-thread branch — both
        // deterministic, since blocking `recv()` here synchronizes on the
        // loop's own observable output instead of racing it.
        assert_eq!(
            next_status(&ui_rx),
            Some(ModelStatus::Downloading {
                received_bytes: 0,
                total_bytes: lock_total_bytes(),
            })
        );
        let failed = next_status(&ui_rx);
        assert!(
            matches!(failed, Some(ModelStatus::Failed { .. })),
            "B2: a download thread that dies without reporting must end in \
             a broadcast Failed status the user can retry from, not \
             silence — got {failed:?}"
        );

        drop(cmd_tx);
        drop(provision_tx);
        handle.join().unwrap();
    }

    // ── B3: two senders into ui_tx let a stale Downloading outlive Failed ─

    /// Direct regression for B3: before the fix, the download thread and a
    /// separate progress-forwarder thread were two independent senders
    /// into the same `ui_tx` — crossbeam gives no ordering guarantee
    /// *across* different senders on one channel, so a `Downloading` tick
    /// already in flight could be delivered *after* a `Failed` that raced
    /// ahead of it, wedging the UI on a frozen progress bar with no way to
    /// retry. Now both are reported *inward* to `run_provisioning_loop`
    /// alone, which retires `progress_rx` the instant a completion is
    /// observed (see the `download_done_rx` arm's comment).
    ///
    /// This queues a stale progress tick and the terminal failure back to
    /// back — both ready for the loop's *next* `select!` call at once,
    /// exactly the race B3 named — and asserts that whichever the loop
    /// happens to service first, the *last* status it ever broadcasts is
    /// the terminal `Failed`, never the stale tick.
    #[test]
    fn a_stale_progress_tick_never_outlives_the_terminal_failed_status() {
        let (progress_tx, progress_rx) = crossbeam_channel::unbounded::<ModelStatus>();
        let (done_tx, done_rx) = crossbeam_channel::bounded::<DownloadOutcome>(1);
        let mut once = Some((done_rx, progress_rx));

        let (cmd_tx, provision_tx, ui_rx, handle) = spawn_loop(
            Phase::NeedsModel(ModelStatus::Missing { total_bytes: 100 }),
            move || once.take().expect("spawn_download called more than once"),
        );

        assert_eq!(
            next_status(&ui_rx),
            Some(ModelStatus::Missing { total_bytes: 100 })
        );

        provision_tx.send(ProvisionCommand::Download).unwrap();
        assert_eq!(
            next_status(&ui_rx),
            Some(ModelStatus::Downloading {
                received_bytes: 0,
                total_bytes: lock_total_bytes(),
            })
        );

        // Both ready at once for the loop's next `select!` call.
        progress_tx
            .send(ModelStatus::Downloading {
                received_bytes: 50,
                total_bytes: lock_total_bytes(),
            })
            .unwrap();
        done_tx
            .send(DownloadOutcome::Failed("connection reset".to_owned()))
            .unwrap();

        let next = next_status(&ui_rx);
        let last = match next {
            Some(ModelStatus::Downloading {
                received_bytes: 50, ..
            }) => next_status(&ui_rx),
            other => other,
        };
        assert!(
            matches!(last, Some(ModelStatus::Failed { .. })),
            "B3: a stale Downloading tick must never be the last status \
             broadcast after a completion — got {last:?}"
        );

        drop(cmd_tx);
        drop(provision_tx);
        handle.join().unwrap();
    }

    // ── on_dictation_command — discard vs. forward ──────────────────────

    #[test]
    fn dictation_commands_are_discarded_before_the_model_is_ready() {
        for phase in [
            Phase::NeedsModel(ModelStatus::Missing { total_bytes: 1 }),
            Phase::Downloading(ModelStatus::Verifying),
            Phase::EngineFailed("boom".to_owned()),
        ] {
            let next = on_dictation_command(phase, DictationCommand::Toggle);
            assert!(
                !matches!(next, Phase::Ready(_)),
                "a Dictation command must never conjure a Ready phase out of \
                 nothing"
            );
        }
    }

    #[test]
    fn dictation_commands_reach_the_session_once_ready() {
        let phase = on_dictation_command(fake_ready_phase(), DictationCommand::Toggle);
        assert!(matches!(phase, Phase::Ready(_)));
    }

    // ── on_provision_command — one test per (Phase, Download) pair ─────

    #[test]
    fn provision_download_from_needs_model_starts_a_download_with_zero_received_bytes() {
        let ui_tx = crossbeam_channel::unbounded().0;
        let event_tx = dummy_event_tx();
        let settings = dummy_settings();
        let mut spawn = || {
            (
                crossbeam_channel::never(),
                crossbeam_channel::never::<ModelStatus>(),
            )
        };

        let (phase, outcome) = on_provision_command(
            Phase::NeedsModel(ModelStatus::Missing { total_bytes: 100 }),
            &ui_tx,
            &event_tx,
            &settings,
            &mut spawn,
        );

        assert_eq!(
            phase_status(&phase),
            ModelStatus::Downloading {
                received_bytes: 0,
                total_bytes: vuho_model_paths::lock().stt.total_bytes,
            },
            "A3: the transition to Downloading must be synchronous with the \
             click, not wait for the first real progress tick"
        );
        assert!(matches!(outcome, ProvisionOutcome::DownloadStarted(..)));
    }

    #[test]
    fn provision_download_while_downloading_is_ignored() {
        let ui_tx = crossbeam_channel::unbounded().0;
        let event_tx = dummy_event_tx();
        let settings = dummy_settings();
        let (phase, outcome) = on_provision_command(
            Phase::Downloading(ModelStatus::Verifying),
            &ui_tx,
            &event_tx,
            &settings,
            &mut unreachable_spawn,
        );
        assert!(matches!(phase, Phase::Downloading(ModelStatus::Verifying)));
        assert!(matches!(outcome, ProvisionOutcome::Ignored));
    }

    #[test]
    fn provision_download_while_ready_is_ignored() {
        let ui_tx = crossbeam_channel::unbounded().0;
        let event_tx = dummy_event_tx();
        let settings = dummy_settings();
        let (phase, outcome) = on_provision_command(
            fake_ready_phase(),
            &ui_tx,
            &event_tx,
            &settings,
            &mut unreachable_spawn,
        );
        assert!(matches!(phase, Phase::Ready(_)));
        assert!(matches!(outcome, ProvisionOutcome::Ignored));
    }

    #[test]
    fn provision_download_while_engine_failed_retries_the_engine_not_the_network() {
        let ui_tx = crossbeam_channel::unbounded().0;
        let event_tx = dummy_event_tx();
        let settings = dummy_settings();
        let (phase, outcome) = on_provision_command(
            Phase::EngineFailed("boom".to_owned()),
            &ui_tx,
            &event_tx,
            &settings,
            &mut unreachable_spawn,
        );
        // `unreachable_spawn` panics if called — merely returning without
        // panicking already proves this phase never starts a download.
        // What `load_engine_and_session` itself does with the retry
        // (whether the real model on this machine loads or not) is out of
        // scope for this unit — it's exercised end-to-end by
        // `test-stt-ffi`; here only the *dispatch* (retry engine, not
        // network) is under test.
        assert!(matches!(outcome, ProvisionOutcome::RetriedBlocking));
        assert!(matches!(phase, Phase::EngineFailed(_) | Phase::Ready(_)));
    }

    // ── on_download_completed ───────────────────────────────────────────

    #[test]
    fn failed_download_outside_downloading_is_left_unchanged() {
        let ui_tx = crossbeam_channel::unbounded().0;
        let event_tx = dummy_event_tx();
        let settings = dummy_settings();
        let phase = on_download_completed(
            Phase::NeedsModel(ModelStatus::Missing { total_bytes: 5 }),
            DownloadOutcome::Failed("simulated".to_owned()),
            &event_tx,
            &ui_tx,
            &settings,
        );
        assert!(matches!(
            phase,
            Phase::NeedsModel(ModelStatus::Missing { total_bytes: 5 })
        ));
    }

    #[test]
    fn failed_download_from_downloading_carries_the_message_into_needs_model() {
        let ui_tx = crossbeam_channel::unbounded().0;
        let event_tx = dummy_event_tx();
        let settings = dummy_settings();
        let phase = on_download_completed(
            Phase::Downloading(ModelStatus::Downloading {
                received_bytes: 10,
                total_bytes: 100,
            }),
            DownloadOutcome::Failed("connection reset".to_owned()),
            &event_tx,
            &ui_tx,
            &settings,
        );
        assert_eq!(
            phase_status(&phase),
            ModelStatus::Failed {
                message: "connection reset".to_owned()
            }
        );
    }
}
