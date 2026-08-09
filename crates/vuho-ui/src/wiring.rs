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
use vuho_dictation::DictationSession;
use vuho_domain::{DictationCommand, DictationEvent, ModelStatus};
use vuho_model_fetch::ModelAvailability;
use vuho_model_paths::Backend;
use vuho_settings::SettingsStore;

use crate::app_state::UiCommand;
use crate::app_status::{HotkeyState, StatusModel};
use crate::event_loop::{spawn_event_drain, spawn_ui_drain};
use crate::panel::PanelRoot;
use crate::settings_tab::SettingsTab;
use crate::{hotkey_presets, permissions, status_bar};

// ── Provisioning state machine (ADR-020) ───────────────────────────────────

/// Command the Settings tab's model rows send to the provisioning thread.
/// `main.rs` owns both ends (CONSTITUTION rule 20): the `Sender` half goes
/// straight into `SettingsTab::new`, the `Receiver` half into
/// [`spawn_warmup_and_bridge`]'s thread alongside `cmd_rx`.
///
/// Each variant names the model it applies to — the Settings tab lists
/// every model the manifest knows, so a command with no id would be
/// ambiguous the moment a second model exists.
pub(crate) enum ProvisionCommand {
    /// Start a download of this model. Also what a `Failed` row's "Retry"
    /// sends — a failed download and a fresh one are the same transition.
    Download(String),
    /// Remove a model Vuho itself downloaded (ADR-020: only
    /// [`vuho_model_paths::ModelSource::UserData`] trees are Vuho's to
    /// delete).
    Delete(String),
    /// Make this model the active one: persist the choice and reload the
    /// engine around it.
    SelectModel(String),
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
/// **Single source of the UI's model state (root-cause fix):** the status
/// that used to be broadcast ad hoc from four different call sites (drifting
/// out of sync with `Phase` itself — the four bugs this replaced) is now
/// *carried by the variant itself* — [`phase_status`] is a pure, total
/// mapping from a `Phase` to the `ModelStatus` it implies, and
/// [`run_provisioning_loop`] is the only place that ever sends
/// `UiCommand::ModelStatus`/`UiCommand::ModelList` (via [`broadcast_phase`]),
/// unconditionally, immediately after every single transition. A `Phase`
/// change without its status reaching the UI is therefore not a bug that can
/// be introduced by a future edit forgetting one of four sites — there is
/// only one site, and it can't be skipped without skipping the transition
/// itself.
enum Phase {
    /// No usable model yet, or a download attempt failed. `id` names the
    /// model `status` is about — the selected one at startup, or whichever
    /// model's download just failed — and `status` carries exactly what the
    /// Settings tab/status bar show, always [`ModelStatus::Missing`] or
    /// [`ModelStatus::Failed`].
    NeedsModel { id: String, status: ModelStatus },
    /// A download is in flight on a second thread, for `id` — which is not
    /// necessarily the selected model: a second model can be downloaded
    /// while the first stays selected. `status` is `Downloading`/`Verifying`,
    /// straight from that thread's progress channel, forwarded into a
    /// `Phase` by [`run_provisioning_loop`]'s `progress_rx` arm. `Dictation`
    /// commands are discarded for as long as this lasts; a `Download` for a
    /// *different* model is refused rather than started, because
    /// [`run_provisioning_loop`]'s `select!` holds exactly one pair of
    /// download receivers — starting a second download would strand the
    /// first with nothing listening to it.
    Downloading { id: String, status: ModelStatus },
    /// `vuho_model_fetch::availability()` reports the selected model is
    /// `Ready`, but the engine itself failed to load (corrupt/incompatible
    /// files, permission error, out of memory, …) — a fact
    /// `Missing`/`Failed` can't express, since the model bytes are actually
    /// fine. `message` is shown the same way a download `Failed` is (a
    /// "Retry" line under the Settings tab's model list — see
    /// [`phase_status`]), and that Retry sends
    /// [`ProvisionCommand::SelectModel`] for the already-selected model,
    /// which re-attempts the engine load only — never a redundant
    /// re-download, which couldn't fix an out-of-band-provisioned model
    /// anyway (ADR-020). Without this variant this state was a dead end:
    /// the model row never showed a way forward, and — the specific bug
    /// this variant fixes — no status ever reached the UI for this branch
    /// at all.
    EngineFailed(String),
    /// Model loaded, engine warmed, session ready to take `DictationCommand`s.
    Ready(vuho_dictation::DictationSession),
}

/// The `ModelStatus` a given [`Phase`] implies — pure and total, so a
/// `Phase` value alone fully determines what the UI shows (see [`Phase`]'s
/// doc comment). The only caller is [`broadcast_phase`].
fn phase_status(phase: &Phase) -> ModelStatus {
    match phase {
        Phase::NeedsModel { status, .. } | Phase::Downloading { status, .. } => status.clone(),
        Phase::EngineFailed(message) => ModelStatus::Failed {
            message: message.clone(),
        },
        Phase::Ready(_) => ModelStatus::Ready,
    }
}

/// The one model whose row the [`Phase`] — not the filesystem — is
/// authoritative about, if any: a download in flight (whose bytes are still
/// under a `.partial` directory `availability()` cannot see) or a model the
/// loop has something to say about that a fresh `availability()` call would
/// not repeat, such as a download that just failed.
fn phase_row_override(phase: &Phase) -> Option<(&str, ModelStatus)> {
    match phase {
        Phase::NeedsModel { id, status } | Phase::Downloading { id, status } => {
            Some((id.as_str(), status.clone()))
        }
        Phase::EngineFailed(_) | Phase::Ready(_) => None,
    }
}

/// The model list to render: `models` as last read from disk, with
/// [`phase_row_override`]'s row replaced.
fn phase_rows(phase: &Phase, models: &[ModelAvailability]) -> Vec<ModelAvailability> {
    let Some((id, status)) = phase_row_override(phase) else {
        return models.to_vec();
    };
    models
        .iter()
        .map(|model| {
            if model.id == id {
                ModelAvailability {
                    status: status.clone(),
                    ..model.clone()
                }
            } else {
                model.clone()
            }
        })
        .collect()
}

/// The **sole** producer of `UiCommand::ModelStatus`/`UiCommand::ModelList`
/// in the process — every call site in this module that can change what
/// [`phase_status`] or [`phase_rows`] report calls this immediately after,
/// with the new value, and nothing else ever sends those commands. See
/// [`Phase`]'s doc comment for why that closes off the four-site drift bug
/// class this state machine replaced.
///
/// The one `phase` assignment that does not broadcast is
/// [`run_provisioning_loop`]'s `cmd_rx` arm: [`on_dictation_command`] hands
/// the command to the session it already holds and returns the same variant
/// carrying the same payload, so there is nothing new for either function to
/// report.
///
/// `phase_status` still reports the *selected* model alone, so the menu-bar
/// composite (`app_status::StatusModel::composite`) is unchanged by the list
/// existing.
fn broadcast_phase(ui_tx: &Sender<UiCommand>, phase: &Phase, models: &[ModelAvailability]) {
    let _ = ui_tx.send(UiCommand::ModelStatus(phase_status(phase)));
    let _ = ui_tx.send(UiCommand::ModelList(phase_rows(phase, models)));
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
        objc2::MainThreadMarker::new().expect(
            "open_dictation_channels runs on the main thread (called from wire_production)",
        ),
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
        let mut load = || load_engine_and_session(&event_tx, &ui_tx, &settings);
        let initial = load_selected_model(&settings, &mut load);
        run_provisioning_loop(
            &cmd_rx,
            &provision_rx,
            &event_tx,
            &ui_tx,
            &settings,
            initial,
            spawn_download_thread,
            load,
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
/// (production: [`load_selected_model`], which touches `vuho_model_fetch` and
/// possibly loads the engine; tests: any `Phase` value, injected directly —
/// this is what makes the loop itself testable without network or a real
/// engine, per this module's test module). `spawn_download` is the
/// side-effecting "start a download thread" step, also injected
/// (CONSTITUTION rule 5): production always passes [`spawn_download_thread`];
/// tests pass a fake under their own control. `load` is the other injected
/// side effect — "load the engine and build a session around it", production
/// [`load_engine_and_session`] — so every transition that reloads the engine
/// is exercisable without a real `CoreML` model.
///
/// Sends `initial`'s status once (see [`broadcast_phase`]) before doing
/// anything else, then drops anything pressed during whatever synchronous
/// work produced it, then services `cmd_rx`/`provision_rx`/download-progress/
/// download-completion messages with `crossbeam_channel::select!` for the
/// rest of the process's life. `download_done_rx`/`progress_rx` start as
/// [`crossbeam_channel::never`] — a receiver that never becomes ready — and
/// are swapped for real ones only while [`Phase::Downloading`], so `select!`
/// never has to special-case "no download in flight" as a `Disconnected`
/// error.
///
/// `models` is the last read of `vuho_model_fetch::availability_all()`,
/// re-read after every transition that can have changed what is on disk. A
/// progress tick is the one message that cannot: it moves bytes only inside
/// the `.partial` directory the resolver never sees, and the one row it does
/// change — the in-flight model's — is supplied by [`phase_row_override`]
/// instead. Re-reading on every tick would re-`stat` every locked file of
/// every model thousands of times during a download for an answer that
/// cannot have moved.
#[allow(
    clippy::too_many_arguments,
    reason = "the loop owns one receiving end per message class it services plus the two \
              injected side effects (CONSTITUTION rule 5) — bundling them into a struct would \
              move the same eight values into a second place with one production call site and \
              one test helper, without removing anything the loop needs"
)]
fn run_provisioning_loop(
    cmd_rx: &Receiver<DictationCommand>,
    provision_rx: &Receiver<ProvisionCommand>,
    event_tx: &Sender<DictationEvent>,
    ui_tx: &Sender<UiCommand>,
    settings: &Arc<SettingsStore>,
    initial: Phase,
    mut spawn_download: impl FnMut(&str) -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>),
    mut load: impl FnMut() -> Option<DictationSession>,
) {
    let mut phase = initial;
    let mut models = vuho_model_fetch::availability_all();
    broadcast_phase(ui_tx, &phase, &models);
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
            recv(provision_rx) -> msg => {
                let Ok(cmd) = msg else {
                    log::info!("provisioning: provision channel disconnected — stopping");
                    return;
                };
                let (next, outcome) = on_provision_command(
                    phase,
                    cmd,
                    event_tx,
                    settings,
                    &mut spawn_download,
                    &mut load,
                );
                phase = next;
                match outcome {
                    ProvisionOutcome::DownloadStarted(done_rx, prog_rx) => {
                        download_done_rx = done_rx;
                        progress_rx = prog_rx;
                    }
                    // A `SelectModel` reload runs the engine load in
                    // place on this thread, blocking it — drain any
                    // presses that landed during that (A1, ADR-007),
                    // same as the startup path above.
                    ProvisionOutcome::ReloadedBlocking => {
                        drain_stale_dictation_commands(cmd_rx);
                    }
                    ProvisionOutcome::Handled => {}
                }
                models = vuho_model_fetch::availability_all();
                broadcast_phase(ui_tx, &phase, &models);
            },
            recv(progress_rx) -> msg => {
                match msg {
                    Ok(status) => {
                        phase = on_download_progress(phase, status);
                        broadcast_phase(ui_tx, &phase, &models);
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
                phase = on_download_completed(phase, outcome, settings, &mut load);
                models = vuho_model_fetch::availability_all();
                broadcast_phase(ui_tx, &phase, &models);
                // Only the `Finished` path runs a blocking engine load
                // (`load_selected_model`) on this thread — a `Failed`
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

/// The model the user picked, resolved against the embedded manifest: an
/// absent setting means "the manifest's default" (ADR-019 keeps the id
/// literal out of `vuho-settings`), and so does a setting naming a model
/// this build no longer ships — a downgrade must fall back to something
/// loadable rather than dead-end on an id nothing can resolve.
///
/// The single place that decision is made (CONSTITUTION rule 26): the
/// provisioning loop and the Settings tab's combobox both read it here.
pub(crate) fn selected_model_id(settings: &SettingsStore) -> String {
    let stt = &vuho_model_paths::manifest().stt;
    match settings.get().speech_model {
        Some(id) if stt.model(&id).is_some() => id,
        Some(unknown) => {
            log::warn!(
                "settings: speech_model {unknown} names no model this build ships — falling back \
                 to {}",
                stt.default_model
            );
            stt.default_model.clone()
        }
        None => stt.default_model.clone(),
    }
}

/// The [`Phase`] the selected model implies right now: refuse to load
/// anything `vuho_model_fetch::availability()` (ADR-020's chokepoint) does
/// not report as `Ready`, or that the running macOS is too old for (WP8.S3),
/// and otherwise load the engine through the injected `load`.
///
/// The one place a load is decided (CONSTITUTION rule 26) — startup, a
/// finished download, and every `SelectModel` all come through here, so
/// "never load a model the chokepoint hasn't cleared" cannot be forgotten at
/// one of three call sites.
fn load_selected_model(
    settings: &SettingsStore,
    load: &mut impl FnMut() -> Option<DictationSession>,
) -> Phase {
    let availability = vuho_model_fetch::availability(&selected_model_id(settings));
    phase_for_availability(availability, load)
}

/// The [`Phase`] one model's [`ModelAvailability`] implies — the decision
/// half of [`load_selected_model`], split from the `availability()` lookup
/// so every refusal is testable without a filesystem in a particular state.
fn phase_for_availability(
    availability: ModelAvailability,
    load: &mut impl FnMut() -> Option<DictationSession>,
) -> Phase {
    if !availability.supported_on_this_os {
        log::warn!(
            "provisioning: {} is not supported by this macOS version",
            availability.id
        );
        return Phase::NeedsModel {
            status: ModelStatus::Failed {
                message: unsupported_message(&availability),
            },
            id: availability.id,
        };
    }
    if availability.status != ModelStatus::Ready {
        log::warn!(
            "provisioning: {} is not ready to load: {:?}",
            availability.id,
            availability.status
        );
        return Phase::NeedsModel {
            id: availability.id,
            status: availability.status,
        };
    }
    match load() {
        Some(session) => Phase::Ready(session),
        None => Phase::EngineFailed(
            "the model is present but the engine failed to load — see the log for details"
                .to_owned(),
        ),
    }
}

/// Why a model this macOS is too old for can't be selected, naming the
/// floor from the manifest rather than a second copy of the version.
fn unsupported_message(availability: &ModelAvailability) -> String {
    match vuho_model_paths::manifest().stt.model(&availability.id) {
        Some(model) => format!(
            "{} needs macOS {} or later.",
            availability.display_name, model.min_macos
        ),
        None => format!(
            "{} is not supported on this Mac.",
            availability.display_name
        ),
    }
}

/// `(Phase, DictationCommand)`: discard while the model isn't ready yet,
/// forward to the session once it is. One of the three exhaustive match
/// sites CONSTITUTION rule 8 requires for this state machine.
fn on_dictation_command(phase: Phase, cmd: DictationCommand) -> Phase {
    match phase {
        p @ (Phase::NeedsModel { .. } | Phase::Downloading { .. } | Phase::EngineFailed(_)) => {
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
/// and "a blocking engine reload just ran on this thread" (needs
/// [`drain_stale_dictation_commands`], A1) can't be confused with each
/// other.
enum ProvisionOutcome {
    /// A download thread is now running: the two receivers
    /// [`run_provisioning_loop`] should install for it.
    DownloadStarted(Receiver<DownloadOutcome>, Receiver<ModelStatus>),
    /// A `SelectModel` ran a blocking engine load on this thread, in place
    /// — win or lose, `cmd_rx` needs draining (ADR-007: a `CapsLock` press
    /// during the reload must not leave the LED on with no session behind
    /// it).
    ReloadedBlocking,
    /// Nothing left for the loop to do: a refused command, or a delete,
    /// which touches the filesystem only. F23: this does *not* close any
    /// window — there is no separate readiness window left to self-close
    /// (ADR-021); the panel (`crate::panel`) simply stays open on whichever
    /// tab the user left it on.
    Handled,
}

/// Total download size of `model_id`, from the repo-pinned lock — the one
/// place this crate reads it, so the progress bar's denominator and the size
/// the download offer quotes can never disagree.
fn model_total_bytes(model_id: &str) -> u64 {
    if let Some(locked) = vuho_model_paths::lock().model(model_id) {
        return locked.total_bytes;
    }
    log::error!(
        "provisioning: {model_id} is absent from models.lock.json — reporting an unknown \
         download size"
    );
    0
}

/// `(Phase, ProvisionCommand)`: the twelve pairs CONSTITUTION rule 8
/// requires, split one function per command so each phase match stays
/// exhaustive and readable — [`on_download_command`], [`on_delete_command`],
/// [`on_select_model_command`].
///
/// `spawn` (start a download thread) and `load` (load the engine and build a
/// session) are the two side effects, injected rather than hardcoded
/// (CONSTITUTION rule 5) so every transition here is testable without the
/// network or a real `CoreML` model.
fn on_provision_command(
    phase: Phase,
    cmd: ProvisionCommand,
    event_tx: &Sender<DictationEvent>,
    settings: &Arc<SettingsStore>,
    spawn: &mut impl FnMut(&str) -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>),
    load: &mut impl FnMut() -> Option<DictationSession>,
) -> (Phase, ProvisionOutcome) {
    match cmd {
        ProvisionCommand::Download(id) => on_download_command(phase, &id, spawn),
        ProvisionCommand::Delete(id) => (
            on_delete_command(phase, &id, event_tx, settings),
            ProvisionOutcome::Handled,
        ),
        ProvisionCommand::SelectModel(id) => on_select_model_command(phase, &id, settings, load),
    }
}

/// `(Phase, Download(id))` — four pairs.
///
/// While [`Phase::Downloading`], a `Download` is refused whichever model it
/// names: the same one is already running, and a *different* one would be
/// orphaned the moment [`run_provisioning_loop`] rebound its single pair of
/// download receivers to the new thread. Every other phase starts the
/// download, moving straight to `Downloading { received_bytes: 0, .. }`
/// rather than waiting for the first progress tick (A3: the row reflects the
/// click immediately, not one network round-trip later).
fn on_download_command(
    phase: Phase,
    id: &str,
    spawn: &mut impl FnMut(&str) -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>),
) -> (Phase, ProvisionOutcome) {
    match phase {
        Phase::Downloading {
            id: in_flight,
            status,
        } => {
            if in_flight == id {
                log::info!("provisioning: {id} is already downloading — ignoring");
            } else {
                log::warn!(
                    "provisioning: refusing to download {id} while {in_flight} is still \
                     downloading"
                );
            }
            (
                Phase::Downloading {
                    id: in_flight,
                    status,
                },
                ProvisionOutcome::Handled,
            )
        }
        phase @ (Phase::NeedsModel { .. } | Phase::EngineFailed(_) | Phase::Ready(_)) => {
            start_download(phase, id, spawn)
        }
    }
}

/// Start a download of `id`, or refuse it with a reason in the log.
///
/// Leaving [`Phase::Ready`] drops the live `DictationSession` — dictation is
/// unavailable for the duration, exactly as it is while the first model
/// downloads, and comes back when [`on_download_completed`] reloads the
/// selected model.
fn start_download(
    phase: Phase,
    id: &str,
    spawn: &mut impl FnMut(&str) -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>),
) -> (Phase, ProvisionOutcome) {
    let availability = vuho_model_fetch::availability(id);
    if vuho_model_paths::manifest().stt.model(id).is_none() {
        log::warn!("provisioning: refusing to download {id} — no such model in the manifest");
        return (phase, ProvisionOutcome::Handled);
    }
    if !availability.supported_on_this_os {
        log::warn!(
            "provisioning: refusing to download {id} — {}",
            unsupported_message(&availability)
        );
        return (phase, ProvisionOutcome::Handled);
    }
    log::info!("provisioning: starting download of {id}");
    let (done_rx, progress_rx) = spawn(id);
    (
        Phase::Downloading {
            id: id.to_owned(),
            status: ModelStatus::Downloading {
                received_bytes: 0,
                total_bytes: model_total_bytes(id),
            },
        },
        ProvisionOutcome::DownloadStarted(done_rx, progress_rx),
    )
}

/// `(Phase, Delete(id))` — four pairs, all leaving the phase untouched:
/// deleting a model can never change the state of the *selected* one,
/// because [`delete_model`] refuses to delete the selected model at all.
fn on_delete_command(
    phase: Phase,
    id: &str,
    event_tx: &Sender<DictationEvent>,
    settings: &Arc<SettingsStore>,
) -> Phase {
    let in_flight = match &phase {
        Phase::Downloading { id, .. } => Some(id.as_str()),
        Phase::NeedsModel { .. } | Phase::EngineFailed(_) | Phase::Ready(_) => None,
    };
    delete_model(id, in_flight, &selected_model_id(settings), event_tx);
    phase
}

/// Delete `id`'s model directory, or refuse with a reason.
///
/// Three refusals, in order: the model whose download is in flight (its
/// bytes are still being written), the selected model (the engine is loaded
/// from it, and deleting it would leave the app with nothing to dictate
/// with), and anything `ModelAvailability::deletable` says is not Vuho's to
/// remove (ADR-020: only trees `vuho_model_fetch::download` itself wrote).
fn delete_model(
    id: &str,
    in_flight: Option<&str>,
    selected: &str,
    event_tx: &Sender<DictationEvent>,
) {
    if in_flight == Some(id) {
        log::warn!("provisioning: refusing to delete {id} — its download is still running");
        return;
    }
    if id == selected {
        log::warn!("provisioning: refusing to delete {id} — it is the selected model");
        return;
    }
    if !vuho_model_fetch::availability(id).deletable() {
        log::warn!("provisioning: refusing to delete {id} — Vuho did not download it");
        return;
    }
    match vuho_model_fetch::delete(id) {
        Ok(()) => log::info!("provisioning: deleted {id}"),
        Err(e) => {
            log::error!("provisioning: failed to delete {id}: {e}");
            let _ = event_tx.send(DictationEvent::Error {
                message: format!("Could not delete {id}: {e}"),
                recoverable: true,
                kind: vuho_domain::ErrorKind::Other,
            });
        }
    }
}

/// `(Phase, SelectModel(id))` — four pairs.
///
/// The choice is persisted first, in every phase, so it survives a restart
/// even when it can't be acted on yet. While [`Phase::Downloading`] that is
/// all that happens: [`on_download_completed`] loads whatever is selected by
/// the time the download finishes, so acting now would mean either
/// abandoning the running download or running two engines at once. Every
/// other phase drops whatever session it holds and reloads through
/// [`load_selected_model`], which refuses a model
/// `vuho_model_fetch::availability` hasn't cleared (WP8.S3) instead of
/// attempting a load that would fail.
fn on_select_model_command(
    phase: Phase,
    id: &str,
    settings: &Arc<SettingsStore>,
    load: &mut impl FnMut() -> Option<DictationSession>,
) -> (Phase, ProvisionOutcome) {
    if vuho_model_paths::manifest().stt.model(id).is_none() {
        log::warn!("provisioning: refusing to select {id} — no such model in the manifest");
        return (phase, ProvisionOutcome::Handled);
    }
    if let Err(e) = settings.update(|s| s.speech_model = Some(id.to_owned())) {
        log::warn!("provisioning: failed to save the speech-model setting: {e}");
    }
    match phase {
        downloading @ Phase::Downloading { .. } => {
            log::info!("provisioning: {id} selected — loading it once the download finishes");
            (downloading, ProvisionOutcome::Handled)
        }
        Phase::Ready(session) => {
            drop(session);
            (
                load_selected_model(settings, load),
                ProvisionOutcome::ReloadedBlocking,
            )
        }
        Phase::NeedsModel { .. } | Phase::EngineFailed(_) => (
            load_selected_model(settings, load),
            ProvisionOutcome::ReloadedBlocking,
        ),
    }
}

/// A progress tick from the download thread. Only [`Phase::Downloading`] can
/// absorb one; a tick observed in any other phase is a straggler from a
/// download that already completed, and applying it would resurrect a
/// `Downloading` state with no download behind it.
fn on_download_progress(phase: Phase, status: ModelStatus) -> Phase {
    match phase {
        Phase::Downloading { id, .. } => Phase::Downloading { id, status },
        other @ (Phase::NeedsModel { .. } | Phase::EngineFailed(_) | Phase::Ready(_)) => {
            log::info!("provisioning: discarding a progress tick from a finished download");
            other
        }
    }
}

/// Download completion, delivered as a message (never a `join()`/sleep —
/// CONSTITUTION rule 32). Only meaningful from [`Phase::Downloading`]; any
/// other phase observing this message would mean a stale completion from an
/// earlier download landed after a state change, so it's passed through
/// unchanged rather than acted on.
///
/// A finished download re-derives readiness through [`load_selected_model`]
/// rather than trusting the download thread's own success signal — the
/// engine must never load until the one chokepoint that decides
/// trustworthiness says `Ready` (ADR-020's enforcement clause) — and it
/// loads the *selected* model, which is not necessarily the one that was
/// downloaded (A `SelectModel` during the download persists only).
fn on_download_completed(
    phase: Phase,
    outcome: DownloadOutcome,
    settings: &Arc<SettingsStore>,
    load: &mut impl FnMut() -> Option<DictationSession>,
) -> Phase {
    match (phase, outcome) {
        (Phase::Downloading { .. }, DownloadOutcome::Finished) => {
            load_selected_model(settings, load)
        }
        (Phase::Downloading { id, .. }, DownloadOutcome::Failed(message)) => {
            log::error!("provisioning: {id} download failed: {message}");
            Phase::NeedsModel {
                id,
                status: ModelStatus::Failed { message },
            }
        }
        (other, _) => other,
    }
}

/// Load the selected model's engine, then build the `DictationSession`
/// around it. [`load_selected_model`] is the only caller, so startup, a
/// finished download, and every `SelectModel` share this one path
/// (CONSTITUTION rule 26).
fn load_engine_and_session(
    event_tx: &Sender<DictationEvent>,
    ui_tx: &Sender<UiCommand>,
    settings: &Arc<SettingsStore>,
) -> Option<DictationSession> {
    log::info!("warmup: loading engine");
    let started = std::time::Instant::now();
    let engine = match load_engine(&selected_model_id(settings)) {
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
    let session = DictationSession::new(event_tx.clone(), engine, settings.clone(), injector);
    let _ = ui_tx.send(UiCommand::EngineReady(Ok(())));
    Some(session)
}

/// Build the engine `model_id`'s manifest entry calls for — the one place a
/// [`Backend`] becomes a concrete engine, so adding a third backend is one
/// match arm here rather than a second load path.
fn load_engine(
    model_id: &str,
) -> Result<Box<dyn vuho_stt_engine::TranscriptionEngine + Send>, vuho_stt_engine::EngineError> {
    let backend = vuho_model_paths::manifest()
        .stt
        .model(model_id)
        .map(|model| model.backend)
        .ok_or_else(|| vuho_stt_engine::EngineError::UnknownModel(model_id.to_owned()))?;
    let folder = vuho_stt_engine::resolve_model_folder(model_id)?;
    Ok(match backend {
        Backend::ParakeetTdt => Box::new(vuho_stt_engine::ParakeetEngine::load(model_id, folder)?),
        Backend::CanaryAed => Box::new(vuho_stt_engine::CanaryEngine::load(model_id, folder)?),
    })
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
/// process (`run_provisioning_loop`'s [`broadcast_phase`]), so that race
/// cannot recur structurally, not just by convention.
fn spawn_download_thread(model_id: &str) -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>) {
    let (done_tx, done_rx) = crossbeam_channel::bounded(1);
    let (progress_tx, progress_rx) = crossbeam_channel::unbounded::<ModelStatus>();
    let model_id = model_id.to_owned();
    std::thread::spawn(move || {
        let outcome = match vuho_model_fetch::download(&model_id, &progress_tx) {
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
/// `permissions::prompt_accessibility` is self-deferring, so calling it here
/// — synchronously, inside GPUI's top-level `Application::run` closure — is
/// safe without this function needing its own deferral wrapper.
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
        permissions::prompt_accessibility();
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

    fn default_model() -> &'static str {
        vuho_model_paths::manifest().stt.default_model.as_str()
    }

    /// `start_download` re-fetches `total_bytes` from the repo-pinned lock
    /// (not from whatever total the `Phase` it is leaving happened to carry)
    /// — this is what a `Downloading{0, ..}` transition's `total_bytes`
    /// actually equals in every test below.
    fn lock_total_bytes() -> u64 {
        model_total_bytes(default_model())
    }

    /// A `spawn_download` fake that is never actually called in a given
    /// test (used where the test only cares about a phase that refuses
    /// `ProvisionCommand::Download`).
    fn unreachable_spawn(_id: &str) -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>) {
        panic!("spawn_download must not be called in this test");
    }

    /// A `load` fake that is never actually called — the falsification
    /// target for every "this transition must not touch the engine" test.
    fn unreachable_load() -> Option<DictationSession> {
        panic!("the engine loader must not be called in this test");
    }

    /// One model's availability, built by hand so a test can state exactly
    /// the readiness it wants without depending on what happens to be on
    /// this machine's disk.
    fn availability_of(id: &str, status: ModelStatus, supported: bool) -> ModelAvailability {
        ModelAvailability {
            id: id.to_owned(),
            display_name: id.to_owned(),
            status,
            source: None,
            total_bytes: 100,
            supported_on_this_os: supported,
        }
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

    fn fake_session() -> DictationSession {
        let injector: vuho_dictation::Injector = Arc::new(|_: &str| Ok(()));
        DictationSession::new(
            dummy_event_tx(),
            Box::new(FakeEngine),
            dummy_settings(),
            injector,
        )
    }

    /// A `Phase::Ready` built from [`FakeEngine`] — for tests of the
    /// `Ready` arm that don't need a real model or a real dictation
    /// session, only a `Phase` value of the right shape.
    fn fake_ready_phase() -> Phase {
        Phase::Ready(fake_session())
    }

    fn needs_model(status: ModelStatus) -> Phase {
        Phase::NeedsModel {
            id: default_model().to_owned(),
            status,
        }
    }

    fn downloading(id: &str, status: ModelStatus) -> Phase {
        Phase::Downloading {
            id: id.to_owned(),
            status,
        }
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
    /// else (`ModelList`, `EngineReady`, `OpenPanel`, …) — what these tests
    /// care about is exactly the one command [`broadcast_phase`] produces
    /// for the selected model.
    fn model_status(cmd: UiCommand) -> Option<ModelStatus> {
        match cmd {
            UiCommand::ModelStatus(status) => Some(status),
            _ => None,
        }
    }

    fn model_list(cmd: UiCommand) -> Option<Vec<ModelAvailability>> {
        match cmd {
            UiCommand::ModelList(models) => Some(models),
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
        F: FnMut(&str) -> (Receiver<DownloadOutcome>, Receiver<ModelStatus>) + Send + 'static,
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
                unreachable_load,
            );
        });
        (cmd_tx, provision_tx, ui_rx, handle)
    }

    // ── selected_model_id — WP8.S1 ──────────────────────────────────────

    #[test]
    fn no_persisted_speech_model_selects_the_manifest_default() {
        let settings = dummy_settings();
        assert_eq!(selected_model_id(&settings), default_model());
    }

    #[test]
    fn an_unknown_persisted_speech_model_falls_back_to_the_manifest_default() {
        let settings = dummy_settings();
        settings
            .update(|s| s.speech_model = Some("a-model-this-build-does-not-ship".to_owned()))
            .expect("temp settings are writable");
        assert_eq!(
            selected_model_id(&settings),
            default_model(),
            "a setting naming a model this build no longer ships must fall back to \
             something loadable, not dead-end on an id nothing can resolve"
        );
    }

    #[test]
    fn a_known_persisted_speech_model_is_selected() {
        let settings = dummy_settings();
        for id in vuho_model_paths::manifest().stt.models.keys() {
            settings
                .update(|s| s.speech_model = Some(id.clone()))
                .expect("temp settings are writable");
            assert_eq!(&selected_model_id(&settings), id);
        }
    }

    // ── phase_for_availability — WP8.S3 ─────────────────────────────────

    #[test]
    fn selecting_a_missing_model_needs_the_model_and_never_loads_an_engine() {
        let phase = phase_for_availability(
            availability_of("some-model", ModelStatus::Missing { total_bytes: 42 }, true),
            &mut unreachable_load,
        );
        match phase {
            Phase::NeedsModel { id, status } => {
                assert_eq!(id, "some-model");
                assert_eq!(status, ModelStatus::Missing { total_bytes: 42 });
            }
            other => panic!("expected NeedsModel, got {:?}", phase_status(&other)),
        }
    }

    #[test]
    fn selecting_a_model_this_macos_is_too_old_for_needs_the_model_and_never_loads_an_engine() {
        let phase = phase_for_availability(
            availability_of("some-model", ModelStatus::Ready, false),
            &mut unreachable_load,
        );
        assert!(
            matches!(phase_status(&phase), ModelStatus::Failed { .. }),
            "an OS-unsupported model must never report Ready — the menu bar would claim a \
             session that cannot exist"
        );
    }

    #[test]
    fn selecting_a_ready_supported_model_loads_the_engine() {
        let phase = phase_for_availability(
            availability_of("some-model", ModelStatus::Ready, true),
            &mut || Some(fake_session()),
        );
        assert!(matches!(phase, Phase::Ready(_)));
    }

    #[test]
    fn a_failing_engine_load_ends_in_engine_failed() {
        let phase = phase_for_availability(
            availability_of("some-model", ModelStatus::Ready, true),
            &mut || None,
        );
        assert!(matches!(phase, Phase::EngineFailed(_)));
    }

    // ── phase_status — pure mapping, one test per variant ──────────────

    #[test]
    fn phase_status_covers_every_variant() {
        assert_eq!(
            phase_status(&needs_model(ModelStatus::Missing { total_bytes: 7 })),
            ModelStatus::Missing { total_bytes: 7 }
        );
        assert_eq!(
            phase_status(&downloading("a", ModelStatus::Verifying)),
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

    // ── phase_rows — the list the Settings tab renders ──────────────────

    #[test]
    fn an_in_flight_download_overrides_only_its_own_row() {
        let models = vec![
            availability_of("a", ModelStatus::Ready, true),
            availability_of("b", ModelStatus::Missing { total_bytes: 100 }, true),
        ];
        let rows = phase_rows(
            &downloading(
                "b",
                ModelStatus::Downloading {
                    received_bytes: 10,
                    total_bytes: 100,
                },
            ),
            &models,
        );
        assert_eq!(rows[0].status, ModelStatus::Ready);
        assert_eq!(
            rows[1].status,
            ModelStatus::Downloading {
                received_bytes: 10,
                total_bytes: 100
            },
            "the downloading model's bytes live under a .partial directory the resolver \
             cannot see — the phase, not the filesystem, is authoritative for that row"
        );
    }

    #[test]
    fn a_ready_phase_leaves_every_row_as_the_filesystem_reported_it() {
        let models = vec![availability_of("a", ModelStatus::Ready, true)];
        assert_eq!(phase_rows(&fake_ready_phase(), &models), models);
    }

    // ── B1: nothing ever produced ModelStatus::Ready ────────────────────

    /// Direct regression for B1: the startup and post-download load paths
    /// used to match `ModelStatus::Ready` in a branch that sent nothing at all —
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
            unreachable_load,
        );

        let commands: Vec<UiCommand> = ui_rx.try_iter().collect();
        let statuses: Vec<ModelStatus> = commands
            .iter()
            .filter_map(|cmd| match cmd {
                UiCommand::ModelStatus(status) => Some(status.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            statuses,
            vec![ModelStatus::Ready],
            "B1: entering Phase::Ready must broadcast ModelStatus::Ready — \
             the bug this regresses sent nothing for the Ready arm at all"
        );
        assert_eq!(
            commands.into_iter().filter_map(model_list).count(),
            1,
            "WP8.S4: every transition broadcasts the model list alongside the status"
        );
    }

    // ── B4: engine-load failure on an available model was a silent sink ─

    /// Direct regression for B4: the startup path's `Ready`-availability/
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
            unreachable_load,
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
            needs_model(ModelStatus::Missing { total_bytes: 100 }),
            |_id| {
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

        provision_tx
            .send(ProvisionCommand::Download(default_model().to_owned()))
            .unwrap();

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
            needs_model(ModelStatus::Missing { total_bytes: 100 }),
            move |_id| once.take().expect("spawn_download called more than once"),
        );

        assert_eq!(
            next_status(&ui_rx),
            Some(ModelStatus::Missing { total_bytes: 100 })
        );

        provision_tx
            .send(ProvisionCommand::Download(default_model().to_owned()))
            .unwrap();
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
            needs_model(ModelStatus::Missing { total_bytes: 1 }),
            downloading("a", ModelStatus::Verifying),
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

    // ── on_download_command — one test per (Phase, Download) pair ───────

    #[test]
    fn download_from_needs_model_starts_a_download_with_zero_received_bytes() {
        let mut spawn = |_id: &str| {
            (
                crossbeam_channel::never(),
                crossbeam_channel::never::<ModelStatus>(),
            )
        };

        let (phase, outcome) = on_download_command(
            needs_model(ModelStatus::Missing { total_bytes: 100 }),
            default_model(),
            &mut spawn,
        );

        assert_eq!(
            phase_status(&phase),
            ModelStatus::Downloading {
                received_bytes: 0,
                total_bytes: lock_total_bytes(),
            },
            "A3: the transition to Downloading must be synchronous with the \
             click, not wait for the first real progress tick"
        );
        assert!(matches!(outcome, ProvisionOutcome::DownloadStarted(..)));
    }

    #[test]
    fn download_from_engine_failed_starts_a_download() {
        let mut spawn = |_id: &str| {
            (
                crossbeam_channel::never(),
                crossbeam_channel::never::<ModelStatus>(),
            )
        };
        let (phase, outcome) = on_download_command(
            Phase::EngineFailed("boom".to_owned()),
            default_model(),
            &mut spawn,
        );
        assert!(matches!(phase, Phase::Downloading { .. }));
        assert!(matches!(outcome, ProvisionOutcome::DownloadStarted(..)));
    }

    #[test]
    fn download_from_ready_starts_a_download() {
        let mut spawn = |_id: &str| {
            (
                crossbeam_channel::never(),
                crossbeam_channel::never::<ModelStatus>(),
            )
        };
        let (phase, outcome) = on_download_command(fake_ready_phase(), default_model(), &mut spawn);
        assert!(matches!(phase, Phase::Downloading { .. }));
        assert!(matches!(outcome, ProvisionOutcome::DownloadStarted(..)));
    }

    #[test]
    fn download_of_the_same_model_while_it_downloads_is_ignored() {
        let (phase, outcome) = on_download_command(
            downloading("a", ModelStatus::Verifying),
            "a",
            &mut unreachable_spawn,
        );
        assert!(matches!(phase, Phase::Downloading { id, .. } if id == "a"));
        assert!(matches!(outcome, ProvisionOutcome::Handled));
    }

    /// `run_provisioning_loop`'s `select!` holds exactly one pair of
    /// download receivers and rebinds both when a download starts — so
    /// starting a second one would leave the first running with nothing
    /// listening to its progress or its completion, wedging the UI on a
    /// download that can never finish as far as the loop is concerned.
    #[test]
    fn download_of_another_model_while_one_is_in_flight_leaves_the_in_flight_one_untouched() {
        let (phase, outcome) = on_download_command(
            downloading(
                "a",
                ModelStatus::Downloading {
                    received_bytes: 10,
                    total_bytes: 100,
                },
            ),
            "b",
            &mut unreachable_spawn, // panics if a second download is started
        );

        match phase {
            Phase::Downloading { id, status } => {
                assert_eq!(id, "a", "the in-flight download must not be replaced");
                assert_eq!(
                    status,
                    ModelStatus::Downloading {
                        received_bytes: 10,
                        total_bytes: 100
                    },
                    "its progress must not be reset either"
                );
            }
            other => panic!(
                "expected the in-flight Downloading phase, got {:?}",
                phase_status(&other)
            ),
        }
        assert!(matches!(outcome, ProvisionOutcome::Handled));
    }

    #[test]
    fn download_of_an_unknown_model_is_refused() {
        let (phase, outcome) = on_download_command(
            needs_model(ModelStatus::Missing { total_bytes: 1 }),
            "no-such-model",
            &mut unreachable_spawn,
        );
        assert!(matches!(phase, Phase::NeedsModel { .. }));
        assert!(matches!(outcome, ProvisionOutcome::Handled));
    }

    // ── on_select_model_command ─────────────────────────────────────────

    #[test]
    fn selecting_a_model_while_downloading_persists_it_without_touching_the_download() {
        let settings = dummy_settings();
        let target = vuho_model_paths::manifest()
            .stt
            .models
            .keys()
            .next()
            .expect("the manifest ships at least one model")
            .clone();

        let (phase, outcome) = on_select_model_command(
            downloading("a", ModelStatus::Verifying),
            &target,
            &settings,
            &mut unreachable_load,
        );

        assert!(matches!(phase, Phase::Downloading { id, .. } if id == "a"));
        assert!(matches!(outcome, ProvisionOutcome::Handled));
        assert_eq!(settings.get().speech_model.as_ref(), Some(&target));
    }

    #[test]
    fn selecting_an_unknown_model_changes_nothing() {
        let settings = dummy_settings();
        let (phase, outcome) = on_select_model_command(
            needs_model(ModelStatus::Missing { total_bytes: 1 }),
            "no-such-model",
            &settings,
            &mut unreachable_load,
        );
        assert!(matches!(phase, Phase::NeedsModel { .. }));
        assert!(matches!(outcome, ProvisionOutcome::Handled));
        assert_eq!(settings.get().speech_model, None);
    }

    // ── on_delete_command ───────────────────────────────────────────────

    #[test]
    fn deleting_the_in_flight_model_is_refused_and_leaves_the_phase_alone() {
        let settings = dummy_settings();
        let event_tx = dummy_event_tx();
        let phase = on_delete_command(
            downloading("a", ModelStatus::Verifying),
            "a",
            &event_tx,
            &settings,
        );
        assert!(matches!(phase, Phase::Downloading { id, .. } if id == "a"));
    }

    #[test]
    fn deleting_the_selected_model_is_refused() {
        let settings = dummy_settings();
        let event_tx = dummy_event_tx();
        let selected = selected_model_id(&settings);
        let phase = on_delete_command(fake_ready_phase(), &selected, &event_tx, &settings);
        assert!(matches!(phase, Phase::Ready(_)));
    }

    #[test]
    fn deleting_from_needs_model_leaves_the_phase_alone() {
        let settings = dummy_settings();
        let event_tx = dummy_event_tx();
        let phase = on_delete_command(
            needs_model(ModelStatus::Missing { total_bytes: 1 }),
            "a-model-this-build-does-not-ship",
            &event_tx,
            &settings,
        );
        assert!(matches!(phase, Phase::NeedsModel { .. }));
    }

    #[test]
    fn deleting_from_engine_failed_leaves_the_phase_alone() {
        let settings = dummy_settings();
        let event_tx = dummy_event_tx();
        let phase = on_delete_command(
            Phase::EngineFailed("boom".to_owned()),
            "a-model-this-build-does-not-ship",
            &event_tx,
            &settings,
        );
        assert!(matches!(phase, Phase::EngineFailed(_)));
    }

    // ── on_download_progress ────────────────────────────────────────────

    #[test]
    fn a_progress_tick_keeps_the_in_flight_model_id() {
        let phase = on_download_progress(
            downloading("a", ModelStatus::Verifying),
            ModelStatus::Downloading {
                received_bytes: 5,
                total_bytes: 10,
            },
        );
        assert!(matches!(phase, Phase::Downloading { id, .. } if id == "a"));
    }

    #[test]
    fn a_progress_tick_after_completion_never_resurrects_downloading() {
        let phase = on_download_progress(
            fake_ready_phase(),
            ModelStatus::Downloading {
                received_bytes: 5,
                total_bytes: 10,
            },
        );
        assert!(matches!(phase, Phase::Ready(_)));
    }

    // ── on_download_completed ───────────────────────────────────────────

    #[test]
    fn failed_download_outside_downloading_is_left_unchanged() {
        let settings = dummy_settings();
        let phase = on_download_completed(
            needs_model(ModelStatus::Missing { total_bytes: 5 }),
            DownloadOutcome::Failed("simulated".to_owned()),
            &settings,
            &mut unreachable_load,
        );
        assert!(matches!(
            phase,
            Phase::NeedsModel {
                status: ModelStatus::Missing { total_bytes: 5 },
                ..
            }
        ));
    }

    #[test]
    fn failed_download_from_downloading_carries_the_message_into_needs_model() {
        let settings = dummy_settings();
        let phase = on_download_completed(
            downloading(
                "a",
                ModelStatus::Downloading {
                    received_bytes: 10,
                    total_bytes: 100,
                },
            ),
            DownloadOutcome::Failed("connection reset".to_owned()),
            &settings,
            &mut unreachable_load,
        );
        assert_eq!(
            phase_status(&phase),
            ModelStatus::Failed {
                message: "connection reset".to_owned()
            }
        );
        assert!(
            matches!(&phase, Phase::NeedsModel { id, .. } if id == "a"),
            "the failed row must be the model that was actually downloading"
        );
    }
}
