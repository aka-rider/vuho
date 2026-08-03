//! macOS menu-bar status item (ADR-006 / GAP 2, ADR-016 gate-mode extension,
//! ADR-020 model-status extension).
//!
//! The app is `LSUIElement` (no Dock icon, accessory activation policy), so the
//! status item is the only always-available way to quit — and, once the app is
//! past the permission gate, to toggle dictation without the `CapsLock` hotkey.
//!
//! The status item must exist in **both** of the app's two modes:
//! - Gate mode ([`install_gate`]): shown while `readiness`'s permission-gate
//!   entry path blocks startup on a missing TCC grant. There is no
//!   dictation session yet, so the menu is just "Permissions…" (re-fronts
//!   the gate window) and "Quit Vuho".
//! - Production mode ([`install`]): the full Start/Stop toggle, Settings,
//!   "Setup…" (re-fronts the readiness window when the speech model isn't
//!   ready — ADR-020), and Quit menu, wired to `wire_production`'s command
//!   channels.
//!
//! One [`StatusDelegate`] class serves both modes (its ivars hold an
//! [`DelegateMode`] enum) so the button configuration, menu-item construction,
//! and the `quit:` action are written exactly once (CONSTITUTION rule 26) —
//! only the menu each mode *assembles* differs.
//!
//! Built with typed objc2-app-kit bindings (like `vuho-os-integration`'s
//! `NSPasteboard` usage). `NSStatusItem`/`NSMenu`/`NSMenuItem` are `MainThreadOnly`,
//! so every call threads a [`MainThreadMarker`]; `install`/`install_gate` must run
//! on the main thread (they are — always called from inside the GPUI `run` closure).
//!
//! All retained `AppKit` objects live in main-thread `thread_local`s, which keeps
//! them alive for the process lifetime and lets [`set_recording`]/[`set_warmup`]
//! mutate the menu label from the GPUI foreground task (same main thread) without
//! any `Send` bound.

use std::cell::RefCell;

use crossbeam_channel::Sender;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol};
use objc2::{define_class, msg_send, sel, AnyThread, DefinedClass, MainThreadMarker};
use objc2_app_kit::{
    NSImage, NSMenu, NSMenuItem, NSStatusBar, NSStatusItem, NSVariableStatusItemLength,
};
use objc2_foundation::NSString;
use vuho_domain::{DictationCommand, ModelStatus};

use crate::app_state::UiCommand;
use crate::readiness::GateCommand;

// ── AppStatus (the one data-driven title mapping — CONSTITUTION rule 26) ──

/// Every state the toggle/status menu item's title can express, across both
/// gate mode and production mode. One enum, one `title()` mapping, instead of
/// separate ad hoc string literals scattered across `set_recording`/
/// `set_warmup`/the gate-mode menu builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AppStatus {
    /// Engine warmup in progress; a press does nothing yet.
    Loading,
    /// Engine ready, not currently recording.
    Ready,
    /// Engine warmup failed; dictation can never start this run.
    EngineFailed,
    /// A dictation session is in progress.
    Recording,
    /// Startup preflight gate is blocking: one or more TCC grants missing.
    PermissionsRequired,
    /// ADR-020: the speech model isn't on disk yet (or a download attempt
    /// failed — see `readiness::model_status_text` for the distinguishing
    /// detail, shown in the readiness window rather than here).
    ModelMissing,
    /// ADR-020: a model download is in progress, `0..=100`.
    Downloading(u8),
    /// ADR-020: the download finished and is being hash-verified.
    Verifying,
}

impl AppStatus {
    fn title(self) -> String {
        match self {
            AppStatus::Loading => "Loading model…".to_owned(),
            AppStatus::Ready => "Start Listening".to_owned(),
            AppStatus::EngineFailed => "Engine unavailable".to_owned(),
            AppStatus::Recording => "Stop Listening".to_owned(),
            AppStatus::PermissionsRequired => "Permissions…".to_owned(),
            AppStatus::ModelMissing => "Model setup needed".to_owned(),
            AppStatus::Downloading(pct) => format!("Downloading model… {pct}%"),
            AppStatus::Verifying => "Verifying model…".to_owned(),
        }
    }
}

/// Map a `vuho_domain::ModelStatus` update to the toggle item's `AppStatus`.
/// `None` for `Ready` — at that point `EngineReady` drives the next title
/// (`Loading model…` → `Start Listening`), not this mapping.
///
/// `Failed` reuses `ModelMissing`'s title: both require the same next user
/// action (open the readiness window and click Download/Retry), and the
/// failure detail itself belongs in that window, not the few characters of
/// menu-bar space here.
fn app_status_for_model(status: &ModelStatus) -> Option<AppStatus> {
    match status {
        ModelStatus::Missing { .. } | ModelStatus::Failed { .. } => Some(AppStatus::ModelMissing),
        ModelStatus::Downloading { .. } => {
            let fraction = status.fraction().unwrap_or(0.0).clamp(0.0, 1.0);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "fraction is clamped to [0.0, 1.0], so *100.0 fits u8 exactly"
            )]
            let pct = (fraction * 100.0).round() as u8;
            Some(AppStatus::Downloading(pct))
        }
        ModelStatus::Verifying => Some(AppStatus::Verifying),
        ModelStatus::Ready => None,
    }
}

/// Reflect a `ModelStatus` update in the toggle item's title, via
/// [`app_status_for_model`]. A no-op for `Ready` and before any install has
/// run (see [`set_toggle_title`]).
pub(crate) fn set_model_status(status: &ModelStatus) {
    if let Some(app_status) = app_status_for_model(status) {
        set_toggle_title(app_status);
    }
}

/// Alias kept for the pre-unification call sites (`spawn_warmup_and_bridge`,
/// `spawn_ui_drain` in `main.rs`) — `AppStatus` covers strictly more states
/// than the old `WarmupState`, so no behavior changes, just the type name.
pub(crate) type WarmupState = AppStatus;

// ── Delegate (one class, two modes) ────────────────────────────────────────

/// What a [`StatusDelegate`] instance can do when its menu items fire.
enum DelegateMode {
    /// Full dictation menu: `cmd_tx` mirrors the hotkey's `DictationCommand`
    /// stream, `ui_tx` reaches the GPUI foreground task for window creation.
    Production {
        cmd_tx: Sender<DictationCommand>,
        ui_tx: Sender<UiCommand>,
    },
    /// Gate mode: no session exists yet, so the only action is re-fronting
    /// (or, per Fix 5, reopening) the permission gate window. `gate_tx`
    /// reaches `readiness::spawn_gate_command_drain`, the GPUI
    /// foreground task that owns gate-window creation — this delegate, like
    /// `Production`'s `ui_tx`, has no `App` access of its own.
    Gate { gate_tx: Sender<GateCommand> },
}

/// Instance variables for the status-item delegate.
struct DelegateIvars {
    mode: DelegateMode,
}

define_class!(
    // The delegate is a plain `NSObject` subclass whose only job is to be the
    // target of the menu items' actions. Every method below is a no-op in
    // whichever mode doesn't apply to it — the corresponding menu item is
    // simply never added to that mode's menu, so it can never fire there.
    #[unsafe(super(NSObject))]
    #[name = "VuhoStatusDelegate"]
    #[ivars = DelegateIvars]
    struct StatusDelegate;

    // SAFETY: `StatusDelegate` is declared above via `define_class!` with
    // `#[unsafe(super(NSObject))]`, which generates a genuine Objective-C
    // subclass of `NSObject` with a matching instance layout and retain/
    // release behavior — the macro, not this `impl`, is what actually
    // upholds `NSObjectProtocol`'s memory-layout contract. Asserting the
    // trait here only tells the type system "this really is an `NSObject`
    // subclass," which is true by construction.
    unsafe impl NSObjectProtocol for StatusDelegate {}

    impl StatusDelegate {
        #[unsafe(method(toggleListening:))]
        fn toggle_listening(&self, _sender: Option<&AnyObject>) {
            if let DelegateMode::Production { cmd_tx, .. } = &self.ivars().mode {
                log::info!("status_bar: menu toggle → Toggle");
                // Best-effort: the bridge/session may already be gone on shutdown.
                let _ = cmd_tx.send(DictationCommand::Toggle);
            }
        }

        #[unsafe(method(openSettings:))]
        fn open_settings(&self, _sender: Option<&AnyObject>) {
            if let DelegateMode::Production { ui_tx, .. } = &self.ivars().mode {
                // Best-effort: the GPUI drain task may already be gone on shutdown.
                let _ = ui_tx.send(UiCommand::OpenSettings);
            }
        }

        #[unsafe(method(showPermissions:))]
        fn show_permissions(&self, _sender: Option<&AnyObject>) {
            if let DelegateMode::Gate { gate_tx } = &self.ivars().mode {
                // Reopens the gate window first if the user closed it,
                // then fronts it (Fix 5) — handled on the GPUI foreground
                // drain (`readiness::spawn_gate_command_drain`) since this
                // AppKit callback has no `App` access of its own.
                // Best-effort: the drain may already be gone on shutdown.
                let _ = gate_tx.send(GateCommand::ReopenOrFront);
            }
        }

        #[unsafe(method(openReadiness:))]
        fn open_readiness(&self, _sender: Option<&AnyObject>) {
            if let DelegateMode::Production { ui_tx, .. } = &self.ivars().mode {
                // Production mode's equivalent of `show_permissions` above:
                // no `gate_tx` exists here (see this module's `DelegateMode`
                // doc comment), so this routes through `UiCommand` and
                // `event_loop::spawn_ui_drain` instead, landing in the same
                // `readiness::reopen_or_front_gate_window` (CONSTITUTION rule 26).
                let _ = ui_tx.send(UiCommand::OpenReadiness);
            }
        }

        #[unsafe(method(quit:))]
        fn quit(&self, _sender: Option<&AnyObject>) {
            // The app's only quit paths: this, and the Cmd+Option+Shift+Q
            // action in main.rs.
            std::process::exit(0);
        }
    }
);

impl StatusDelegate {
    fn new(mode: DelegateMode) -> Retained<Self> {
        let this = Self::alloc().set_ivars(DelegateIvars { mode });
        // SAFETY: `this` is a freshly allocated, ivar-initialized
        // `StatusDelegate` from `alloc().set_ivars(..)` — calling
        // `[super init]` on it exactly once is the standard, required
        // Objective-C two-phase (`alloc` + `init`) construction sequence
        // `define_class!`-generated subclasses must follow.
        unsafe { msg_send![super(this), init] }
    }
}

// ── Shared item/menu construction ──────────────────────────────────────────

/// Retained `AppKit` objects for a status item, held for the app lifetime.
/// Shared shape for both gate-mode and production installs; `toggle_item` is
/// `None` in gate mode (its single "Permissions…" item never changes title).
struct StatusState {
    _item: Retained<NSStatusItem>,
    _menu: Retained<NSMenu>,
    toggle_item: Option<Retained<NSMenuItem>>,
    _delegate: Retained<StatusDelegate>,
}

thread_local! {
    /// Main-thread-only storage for whichever status item is currently
    /// installed (gate-mode xor production — `main.rs` installs exactly one
    /// per process, never both). Populated by [`install`]/[`install_gate`],
    /// read by [`set_recording`]/[`set_warmup`]; all run on the GPUI main thread.
    static STATE: RefCell<Option<StatusState>> = const { RefCell::new(None) };
}

/// Create the status item + button icon + delegate; the caller builds and
/// attaches the mode-specific menu. Common to [`install`] and [`install_gate`]
/// (CONSTITUTION rule 26 — one construction path, not two).
fn new_status_item(
    mode: DelegateMode,
    mtm: MainThreadMarker,
) -> (Retained<NSStatusItem>, Retained<StatusDelegate>) {
    let delegate = StatusDelegate::new(mode);
    let bar = NSStatusBar::systemStatusBar();
    let item = bar.statusItemWithLength(NSVariableStatusItemLength);
    configure_status_button(&item, mtm);
    (item, delegate)
}

/// Install the menu-bar status item with a Start/Stop toggle, Settings, and Quit.
///
/// Must be called on the main thread (from the GPUI `run` closure). Menu clicks
/// send [`DictationCommand::Toggle`] on `cmd_tx` — the same channel the `CapsLock`
/// hotkey uses — and [`UiCommand::OpenSettings`] on `ui_tx`, drained by the
/// GPUI foreground task that owns window creation.
pub(crate) fn install(cmd_tx: Sender<DictationCommand>, ui_tx: Sender<UiCommand>) {
    let Some(mtm) = MainThreadMarker::new() else {
        log::error!("status_bar::install must be called on the main thread");
        return;
    };

    let (item, delegate) = new_status_item(DelegateMode::Production { cmd_tx, ui_tx }, mtm);
    let (menu, toggle_item) = build_menu(&delegate, mtm);
    item.setMenu(Some(&menu));

    STATE.with(|s| {
        *s.borrow_mut() = Some(StatusState {
            _item: item,
            _menu: menu,
            toggle_item: Some(toggle_item),
            _delegate: delegate,
        });
    });
}

/// Install the gate-mode status item: "Permissions…" (reopens/re-fronts the
/// gate window, see [`crate::readiness::reopen_or_front_gate_window`])
/// · separator · "Quit Vuho". Called instead of [`install`] while
/// `readiness::missing_permissions()` is non-empty, so the app is never
/// silently running with no menu-bar affordance at all (Fix 2).
pub(crate) fn install_gate(gate_tx: Sender<GateCommand>) {
    let Some(mtm) = MainThreadMarker::new() else {
        log::error!("status_bar::install_gate must be called on the main thread");
        return;
    };

    let (item, delegate) = new_status_item(DelegateMode::Gate { gate_tx }, mtm);
    let menu = build_gate_menu(&delegate, mtm);
    item.setMenu(Some(&menu));

    STATE.with(|s| {
        *s.borrow_mut() = Some(StatusState {
            _item: item,
            _menu: menu,
            toggle_item: None,
            _delegate: delegate,
        });
    });
}

/// Set the status-item button's icon: an SF Symbol template image, with a
/// plain-text fallback if the symbol is unavailable. Shared by both modes —
/// the waveform icon is how the user recognizes Vuho is running at all,
/// gate or production.
fn configure_status_button(item: &NSStatusItem, mtm: MainThreadMarker) {
    let Some(button) = item.button(mtm) else {
        return;
    };
    if let Some(image) = NSImage::imageWithSystemSymbolName_accessibilityDescription(
        &NSString::from_str("waveform"),
        Some(&NSString::from_str("Vuho")),
    ) {
        image.setTemplate(true);
        button.setImage(Some(&image));
    } else {
        button.setTitle(&NSString::from_str("𝗏"));
    }
}

/// Build one menu item: title + action selector + key equivalent, targeted
/// at `target`, and add it to `menu`. Returns the retained item so callers
/// that need to keep a handle (the toggle item's title flips at runtime) can
/// hold onto it.
fn make_menu_item(
    mtm: MainThreadMarker,
    menu: &NSMenu,
    title: &str,
    action: objc2::runtime::Sel,
    key_equivalent: &str,
    target: &AnyObject,
) -> Retained<NSMenuItem> {
    // SAFETY: `mtm.alloc::<NSMenuItem>()` produces a freshly allocated,
    // uninitialized `NSMenuItem` — `initWithTitle_action_keyEquivalent`'s
    // contract (like every Objective-C `init...` method) is to consume that
    // allocation exactly once and return the (possibly different, per
    // Cocoa's `init` convention) initialized instance; calling it here on
    // the value `alloc` just produced, exactly once, satisfies that.
    let item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            mtm.alloc::<NSMenuItem>(),
            &NSString::from_str(title),
            Some(action),
            &NSString::from_str(key_equivalent),
        )
    };
    // SAFETY: `setTarget:` stores an unretained (weak, Cocoa-convention)
    // reference to `target` on `item`. `target` is `&StatusDelegate`'s
    // `Retained` handle held alive for the process lifetime in
    // `StatusState`/`STATE` (see the thread_local doc comment below), which
    // outlives every menu item built from it — no dangling-target risk.
    unsafe { item.setTarget(Some(target)) };
    menu.addItem(&item);
    item
}

/// Build the "Start Listening" (toggle) · separator · "Settings…" ·
/// separator · "Quit Vuho" menu.
fn build_menu(
    delegate: &Retained<StatusDelegate>,
    mtm: MainThreadMarker,
) -> (Retained<NSMenu>, Retained<NSMenuItem>) {
    let menu = NSMenu::new(mtm);
    let delegate_target: &AnyObject = delegate;

    let toggle_item = make_menu_item(
        mtm,
        &menu,
        &AppStatus::Loading.title(),
        sel!(toggleListening:),
        "",
        delegate_target,
    );

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    make_menu_item(
        mtm,
        &menu,
        "Settings…",
        sel!(openSettings:),
        ",",
        delegate_target,
    );

    // ADR-020: re-front (or reopen) the readiness window — the production-mode
    // path's equivalent of gate mode's "Permissions…" item (`build_gate_menu`),
    // routed through `UiCommand::OpenReadiness` instead of `GateCommand`
    // (see `DelegateMode`'s doc comment for why production mode can't use
    // the same channel type).
    make_menu_item(
        mtm,
        &menu,
        "Setup…",
        sel!(openReadiness:),
        "",
        delegate_target,
    );

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    make_menu_item(mtm, &menu, "Quit Vuho", sel!(quit:), "q", delegate_target);

    (menu, toggle_item)
}

/// Build the gate-mode "Permissions…" · separator · "Quit Vuho" menu.
fn build_gate_menu(delegate: &Retained<StatusDelegate>, mtm: MainThreadMarker) -> Retained<NSMenu> {
    let menu = NSMenu::new(mtm);
    let delegate_target: &AnyObject = delegate;

    make_menu_item(
        mtm,
        &menu,
        &AppStatus::PermissionsRequired.title(),
        sel!(showPermissions:),
        "",
        delegate_target,
    );

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    make_menu_item(mtm, &menu, "Quit Vuho", sel!(quit:), "q", delegate_target);

    menu
}

/// Update the toggle item's title to reflect recording state.
///
/// Called from the GPUI foreground drain task (main thread) — same thread as
/// [`install`], so it sees the populated `thread_local`. A no-op in gate mode
/// (no `toggle_item`) or before any install has run.
pub(crate) fn set_recording(recording: bool) {
    set_toggle_title(if recording {
        AppStatus::Recording
    } else {
        AppStatus::Ready
    });
}

/// Reflect engine warmup state in the toggle item.
///
/// The model load takes minutes on a cold ANE cache; without this the menu
/// would read "Start Listening" while a press could not possibly start
/// anything.
pub(crate) fn set_warmup(state: AppStatus) {
    set_toggle_title(state);
}

fn set_toggle_title(status: AppStatus) {
    STATE.with(|s| {
        if let Some(Some(item)) = s.borrow().as_ref().map(|state| &state.toggle_item) {
            item.setTitle(&NSString::from_str(&status.title()));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_status_titles_are_distinct_and_nonempty() {
        let all = [
            AppStatus::Loading,
            AppStatus::Ready,
            AppStatus::EngineFailed,
            AppStatus::Recording,
            AppStatus::PermissionsRequired,
            AppStatus::ModelMissing,
            AppStatus::Downloading(43),
            AppStatus::Verifying,
        ];
        let titles: Vec<String> = all.iter().map(|s| s.title()).collect();
        for title in &titles {
            assert!(!title.is_empty());
        }
        let mut unique = titles.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            titles.len(),
            "titles must be distinct: {titles:?}"
        );
    }

    #[test]
    fn app_status_permissions_required_title() {
        assert_eq!(AppStatus::PermissionsRequired.title(), "Permissions…");
    }

    #[test]
    fn app_status_recording_vs_ready_titles_match_toggle_semantics() {
        assert_eq!(AppStatus::Recording.title(), "Stop Listening");
        assert_eq!(AppStatus::Ready.title(), "Start Listening");
    }

    #[test]
    fn app_status_downloading_title_embeds_the_percentage() {
        assert_eq!(AppStatus::Downloading(43).title(), "Downloading model… 43%");
    }

    // ── app_status_for_model — pure, no GPUI/AppKit involved ──────────────

    #[test]
    fn app_status_for_model_ready_is_none() {
        assert_eq!(app_status_for_model(&ModelStatus::Ready), None);
    }

    #[test]
    fn app_status_for_model_missing_and_failed_both_map_to_model_missing() {
        assert_eq!(
            app_status_for_model(&ModelStatus::Missing { total_bytes: 100 }),
            Some(AppStatus::ModelMissing)
        );
        assert_eq!(
            app_status_for_model(&ModelStatus::Failed {
                message: "oops".to_owned()
            }),
            Some(AppStatus::ModelMissing)
        );
    }

    #[test]
    fn app_status_for_model_verifying_maps_to_verifying() {
        assert_eq!(
            app_status_for_model(&ModelStatus::Verifying),
            Some(AppStatus::Verifying)
        );
    }

    #[test]
    fn app_status_for_model_downloading_rounds_the_percentage() {
        assert_eq!(
            app_status_for_model(&ModelStatus::Downloading {
                received_bytes: 43,
                total_bytes: 100,
            }),
            Some(AppStatus::Downloading(43))
        );
        // Rounds, not truncates: 254/1000 = 25.4% → 25%.
        assert_eq!(
            app_status_for_model(&ModelStatus::Downloading {
                received_bytes: 254,
                total_bytes: 1_000,
            }),
            Some(AppStatus::Downloading(25))
        );
    }
}
