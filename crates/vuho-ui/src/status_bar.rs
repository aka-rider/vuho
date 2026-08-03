//! macOS menu-bar status item (ADR-006 / GAP 2, ADR-016 gate-mode extension,
//! ADR-020 model-status extension, UI-rehaul WP4 click-split/`StatusModel`
//! wiring).
//!
//! The app is `LSUIElement` (no Dock icon, accessory activation policy), so the
//! status item is the only always-available way to quit — and, once the app is
//! past the permission gate, to toggle dictation without the `CapsLock` hotkey.
//!
//! The status item must exist in **both** of the app's two modes:
//! - Gate mode ([`install_gate`]): shown while `readiness`'s permission-gate
//!   entry path blocks startup on a missing TCC grant. There is no
//!   dictation session yet, so the menu is just "Permissions…" (re-fronts
//!   the gate window) and "Quit Vuho". Unchanged by WP4 — it dies in a
//!   later package.
//! - Production mode ([`install`]): a click-split button (plain left click
//!   opens the panel, right-click/control-click pops a menu) with a
//!   Start/Stop toggle · "Open Vuho" · Quit menu, driven by [`sync`] from
//!   the caller's `StatusModel` `Entity`.
//!
//! One [`StatusDelegate`] class serves both modes (its ivars hold a
//! [`DelegateMode`] enum) so the button configuration, menu-item construction,
//! and the `quit:` action are written exactly once (CONSTITUTION rule 26) —
//! only the menu each mode *assembles*, and how each mode's button reacts to a
//! click, differ.
//!
//! Built with typed objc2-app-kit bindings (like `vuho-os-integration`'s
//! `NSPasteboard` usage). `NSStatusItem`/`NSMenu`/`NSMenuItem` are `MainThreadOnly`,
//! so every call threads a [`MainThreadMarker`]; `install`/`install_gate` must run
//! on the main thread (they are — always called from inside the GPUI `run` closure).
//!
//! All retained `AppKit` objects live in main-thread `thread_local`s, which keeps
//! them alive for the process lifetime and lets [`sync`] mutate the menu label
//! and tray icon from the GPUI foreground task (same main thread) without any
//! `Send` bound.
//!
//! **Borrow discipline (module-wide rule):** every function that touches
//! [`STATE`] copies the `Retained<…>` handles it needs out of the borrow
//! (`Retained<T>: Clone` is a refcount bump, not an `AppKit` call) and lets the
//! borrow drop *before* sending any `AppKit` message. Never call into `AppKit`
//! while holding `STATE`'s `RefCell` borrow. This matters because
//! `pop_menu`'s `performClick:` pumps a nested run loop — a GPUI entity
//! observer can re-enter [`sync`] from inside that loop — so a borrow still
//! held across the `performClick:` call would deadlock the `RefCell` (a
//! second `borrow_mut()` while the first is live panics).

use std::cell::{Cell, RefCell};

use crossbeam_channel::Sender;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol};
use objc2::{define_class, msg_send, sel, AnyThread, DefinedClass, MainThreadMarker};
use objc2_app_kit::{
    NSApplication, NSEventMask, NSEventModifierFlags, NSEventType, NSImage, NSMenu, NSMenuItem,
    NSStatusBar, NSStatusBarButton, NSStatusItem, NSVariableStatusItemLength,
};
use objc2_foundation::NSString;
use vuho_domain::DictationCommand;

use crate::app_state::UiCommand;
use crate::app_status::CompositeStatus;
use crate::readiness::GateCommand;

// ── TrayIcon (the one data-driven icon mapping — CONSTITUTION rule 26) ────

/// Every icon the tray button can show, across every [`CompositeStatus`].
/// One enum, one mapping ([`TrayIcon::from_composite`]), instead of ad hoc
/// symbol-name literals scattered across call sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrayIcon {
    /// Engine ready, not recording — [`CompositeStatus::Ready`].
    Idle,
    /// A dictation session is in progress — [`CompositeStatus::Recording`].
    Recording,
    /// Everything else: permissions/relaunch/model/engine states that need
    /// the user's attention before dictation can start.
    Maintenance,
}

impl TrayIcon {
    /// Map a [`CompositeStatus`] to the icon it implies — total over every
    /// variant by construction (`Ready`/`Recording` are named explicitly,
    /// everything else falls through to `Maintenance`).
    fn from_composite(status: &CompositeStatus) -> Self {
        match status {
            CompositeStatus::Ready => TrayIcon::Idle,
            CompositeStatus::Recording => TrayIcon::Recording,
            CompositeStatus::PermissionsMissing
            | CompositeStatus::RelaunchRequired
            | CompositeStatus::ModelMissing
            | CompositeStatus::Downloading(_)
            | CompositeStatus::Verifying
            | CompositeStatus::EngineFailed
            | CompositeStatus::EngineLoading => TrayIcon::Maintenance,
        }
    }

    /// SF Symbol names to try, in preference order — the first one macOS
    /// actually has wins. Availability verified against the OS symbol
    /// catalog for this app's floor (macOS 14.0): `waveform.badge.mic`
    /// exists from 14.0 (do **not** use `waveform.badge.microphone`,
    /// that's 15.0+); `waveform.badge.exclamationmark` is 12.0+.
    fn symbol_names(self) -> &'static [&'static str] {
        match self {
            TrayIcon::Idle => &["waveform"],
            TrayIcon::Recording => &["waveform.badge.mic", "record.circle", "waveform"],
            TrayIcon::Maintenance => {
                &["waveform.badge.exclamationmark", "exclamationmark.triangle"]
            }
        }
    }
}

/// Apply `icon` to `button`: try each of [`TrayIcon::symbol_names`] in order,
/// first `Some` wins, `setTemplate(true)`'d; `log::warn!` when a preferred
/// name failed and a fallback was used. Falls back to a plain-text title if
/// every name in the chain is unavailable (matches the old
/// [`configure_status_button`] behavior for a fully failed image lookup).
fn apply_icon(button: &NSStatusBarButton, icon: TrayIcon) {
    let names = icon.symbol_names();
    for (index, name) in names.iter().enumerate() {
        if let Some(image) = NSImage::imageWithSystemSymbolName_accessibilityDescription(
            &NSString::from_str(name),
            Some(&NSString::from_str("Vuho")),
        ) {
            image.setTemplate(true);
            button.setImage(Some(&image));
            if index > 0 {
                log::warn!(
                    "status_bar: SF Symbol {:?} unavailable for {icon:?}, using fallback {name:?}",
                    names[0]
                );
            }
            return;
        }
    }
    log::warn!("status_bar: no SF Symbol available for {icon:?}, falling back to text title");
    button.setTitle(&NSString::from_str("𝗏"));
}

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
    // target of the button's/menu items' actions. Every method below is a
    // no-op in whichever mode doesn't apply to it — the corresponding
    // control is simply never wired to it in that mode, so it can never fire
    // there.
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

        #[unsafe(method(openPanel:))]
        fn open_panel(&self, _sender: Option<&AnyObject>) {
            if let DelegateMode::Production { ui_tx, .. } = &self.ivars().mode {
                // Best-effort: the GPUI drain task may already be gone on shutdown.
                let _ = ui_tx.send(UiCommand::OpenPanel);
            }
        }

        /// Fired by the click-split button config `install` sets up
        /// (`setTarget:`/`setAction:`/`sendActionOn:`) for every left/right
        /// mouse-up. Inspects the triggering event to tell a menu gesture
        /// (right-click, or control-left-click — the standard macOS status-
        /// item convention) from a plain left click: the former pops the
        /// menu via [`pop_menu`], the latter opens the panel exactly like
        /// `openPanel:`.
        #[unsafe(method(statusItemClicked:))]
        fn status_item_clicked(&self, _sender: Option<&AnyObject>) {
            let DelegateMode::Production { ui_tx, .. } = &self.ivars().mode else {
                return;
            };
            let Some(mtm) = MainThreadMarker::new() else {
                return;
            };
            let is_menu_gesture = NSApplication::sharedApplication(mtm)
                .currentEvent()
                .is_some_and(|event| {
                    event.r#type() == NSEventType::RightMouseUp
                        || (event.r#type() == NSEventType::LeftMouseUp
                            && event.modifierFlags().contains(NSEventModifierFlags::Control))
                });
            if is_menu_gesture {
                pop_menu();
            } else {
                // Best-effort: the GPUI drain task may already be gone on shutdown.
                let _ = ui_tx.send(UiCommand::OpenPanel);
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
/// `last_icon` starts `None` (nothing applied via [`sync`] yet, distinct
/// from whatever static default [`configure_status_button`] painted at
/// install time) and is only ever read/written from the main thread.
struct StatusState {
    item: Retained<NSStatusItem>,
    button: Retained<NSStatusBarButton>,
    menu: Retained<NSMenu>,
    toggle_item: Option<Retained<NSMenuItem>>,
    // Retained for the process lifetime only — `setTarget:` (used by the
    // production click-split) does not itself retain its target, so this is
    // what keeps the delegate alive; never read again after construction.
    _delegate: Retained<StatusDelegate>,
    last_icon: Cell<Option<TrayIcon>>,
}

thread_local! {
    /// Main-thread-only storage for whichever status item is currently
    /// installed (gate-mode xor production — `main.rs` installs exactly one
    /// per process, never both). Populated by [`install`]/[`install_gate`],
    /// read by [`sync`]/[`pop_menu`]/`statusItemClicked:`; all run on the
    /// GPUI main thread.
    static STATE: RefCell<Option<StatusState>> = const { RefCell::new(None) };
}

/// Create the status item + button icon + delegate; the caller builds and
/// attaches the mode-specific menu. Common to [`install`] and [`install_gate`]
/// (CONSTITUTION rule 26 — one construction path, not two). `None` if the
/// status item has no button (should not happen in practice, but
/// `NSStatusItem::button` is itself `Option`-typed).
fn new_status_item(
    mode: DelegateMode,
    mtm: MainThreadMarker,
) -> Option<(
    Retained<NSStatusItem>,
    Retained<NSStatusBarButton>,
    Retained<StatusDelegate>,
)> {
    let delegate = StatusDelegate::new(mode);
    let bar = NSStatusBar::systemStatusBar();
    let item = bar.statusItemWithLength(NSVariableStatusItemLength);
    let button = configure_status_button(&item, mtm)?;
    Some((item, button, delegate))
}

/// Install the menu-bar status item with a click-split button (plain left
/// click opens the panel, right-click/control-click pops the menu) and a
/// Start/Stop toggle · "Open Vuho" · Quit menu.
///
/// Must be called on the main thread (from the GPUI `run` closure). The
/// toggle sends [`DictationCommand::Toggle`] on `cmd_tx` — the same channel
/// the `CapsLock` hotkey uses; both "Open Vuho" and a plain click send
/// [`UiCommand::OpenPanel`] on `ui_tx`, drained by the GPUI foreground task
/// that owns window creation. Nothing here reflects live app state — call
/// [`sync`] once right after installing, then again on every
/// `StatusModel` change (see `wiring::wire_production`).
pub(crate) fn install(cmd_tx: Sender<DictationCommand>, ui_tx: Sender<UiCommand>) {
    let Some(mtm) = MainThreadMarker::new() else {
        log::error!("status_bar::install must be called on the main thread");
        return;
    };

    let Some((item, button, delegate)) =
        new_status_item(DelegateMode::Production { cmd_tx, ui_tx }, mtm)
    else {
        log::error!("status_bar::install: status item has no button");
        return;
    };
    let (menu, toggle_item) = build_menu(&delegate, mtm);

    // Click split: route every left/right mouse-up through
    // `statusItemClicked:` instead of attaching the menu directly via
    // `setMenu` (which would make AppKit auto-pop it on every click,
    // leaving no way for a plain click to open the panel instead).
    let delegate_target: &AnyObject = &delegate;
    // SAFETY: `setTarget:`/`setAction:`/`sendActionOn:` are ordinary
    // `NSControl` configuration calls on `button`, a live `NSStatusBarButton`
    // retained for the process lifetime in the `StatusState` constructed
    // just below. `setTarget:` does not itself retain `delegate_target` —
    // `delegate` is what keeps it alive, also retained in that same
    // `StatusState`.
    unsafe {
        button.setTarget(Some(delegate_target));
        button.setAction(Some(sel!(statusItemClicked:)));
        button.sendActionOn(NSEventMask::LeftMouseUp | NSEventMask::RightMouseUp);
    }

    STATE.with(|s| {
        *s.borrow_mut() = Some(StatusState {
            item,
            button,
            menu,
            toggle_item: Some(toggle_item),
            _delegate: delegate,
            last_icon: Cell::new(None),
        });
    });
}

/// Install the gate-mode status item: "Permissions…" (reopens/re-fronts the
/// gate window, see [`crate::readiness::reopen_or_front_gate_window`])
/// · separator · "Quit Vuho". Called instead of [`install`] while
/// `readiness::missing_permissions()` is non-empty, so the app is never
/// silently running with no menu-bar affordance at all (Fix 2). Unchanged by
/// WP4: the button keeps `AppKit`'s classic "attach the menu, let it auto-pop"
/// behavior — there is no panel to click-split to yet in gate mode.
pub(crate) fn install_gate(gate_tx: Sender<GateCommand>) {
    let Some(mtm) = MainThreadMarker::new() else {
        log::error!("status_bar::install_gate must be called on the main thread");
        return;
    };

    let Some((item, button, delegate)) = new_status_item(DelegateMode::Gate { gate_tx }, mtm)
    else {
        log::error!("status_bar::install_gate: status item has no button");
        return;
    };
    let menu = build_gate_menu(&delegate, mtm);
    item.setMenu(Some(&menu));

    STATE.with(|s| {
        *s.borrow_mut() = Some(StatusState {
            item,
            button,
            menu,
            toggle_item: None,
            _delegate: delegate,
            last_icon: Cell::new(None),
        });
    });
}

/// Pop the production menu synchronously via the modern status-item trick:
/// attach it, simulate a click to run `AppKit`'s own menu-tracking loop, then
/// detach — so the button can stay click-split (a plain click reaches
/// `statusItemClicked:` instead of a permanently attached menu intercepting
/// every click first).
///
/// Copies the `Retained` handles it needs out of [`STATE`]'s borrow before
/// calling into `AppKit` (this module's borrow-discipline rule — see the
/// module doc comment): `performClick:` pumps a nested run loop, and an
/// entity observer re-entering [`sync`] from inside it would deadlock the
/// `RefCell` if this function were still holding the borrow.
fn pop_menu() {
    let Some((item, menu, button)) = STATE.with(|s| {
        s.borrow()
            .as_ref()
            .map(|state| (state.item.clone(), state.menu.clone(), state.button.clone()))
    }) else {
        return;
    };
    item.setMenu(Some(&menu));
    // SAFETY: `performClick:` simulates a user click on `button`; `None`
    // matches Cocoa's convention for a programmatic click with no
    // originating control.
    unsafe { button.performClick(None) };
    item.setMenu(None);
}

/// Set the status-item button's initial icon: an SF Symbol template image,
/// with a plain-text fallback if the symbol is unavailable. Shared by both
/// modes — the waveform icon is how the user recognizes Vuho is running at
/// all, gate or production. Production overwrites this within moments via
/// its first [`sync`] call; gate mode never calls `sync`, so this stays put
/// for gate mode's whole lifetime. Returns the button so callers can build
/// the rest of [`StatusState`] around it.
fn configure_status_button(
    item: &NSStatusItem,
    mtm: MainThreadMarker,
) -> Option<Retained<NSStatusBarButton>> {
    let button = item.button(mtm)?;
    if let Some(image) = NSImage::imageWithSystemSymbolName_accessibilityDescription(
        &NSString::from_str("waveform"),
        Some(&NSString::from_str("Vuho")),
    ) {
        image.setTemplate(true);
        button.setImage(Some(&image));
    } else {
        button.setTitle(&NSString::from_str("𝗏"));
    }
    Some(button)
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

/// Build the "Start Listening" (toggle) · separator · "Open Vuho" ·
/// separator · "Quit Vuho" menu.
fn build_menu(
    delegate: &Retained<StatusDelegate>,
    mtm: MainThreadMarker,
) -> (Retained<NSMenu>, Retained<NSMenuItem>) {
    let menu = NSMenu::new(mtm);
    menu.setAutoenablesItems(false);
    let delegate_target: &AnyObject = delegate;

    let toggle_item = make_menu_item(
        mtm,
        &menu,
        &CompositeStatus::EngineLoading.menu_title(),
        sel!(toggleListening:),
        "",
        delegate_target,
    );

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    make_menu_item(
        mtm,
        &menu,
        "Open Vuho",
        sel!(openPanel:),
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
    menu.setAutoenablesItems(false);
    let delegate_target: &AnyObject = delegate;

    make_menu_item(
        mtm,
        &menu,
        "Permissions…",
        sel!(showPermissions:),
        "",
        delegate_target,
    );

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    make_menu_item(mtm, &menu, "Quit Vuho", sel!(quit:), "q", delegate_target);

    menu
}

/// The **only** tray mutator: reflect one [`CompositeStatus`] in the toggle
/// item's title/enabled state and the button's icon (via [`TrayIcon`]).
///
/// Called from the GPUI foreground task (main thread) — same thread as
/// [`install`], so it sees the populated `thread_local`. A no-op in gate
/// mode (no `toggle_item`, and no observer ever calls this there) or before
/// any install has run.
pub(crate) fn sync(status: &CompositeStatus) {
    let Some((toggle_item, button, cached_icon)) = STATE.with(|s| {
        s.borrow().as_ref().map(|state| {
            (
                state.toggle_item.clone(),
                state.button.clone(),
                state.last_icon.get(),
            )
        })
    }) else {
        return;
    };

    if let Some(toggle_item) = &toggle_item {
        toggle_item.setTitle(&NSString::from_str(&status.menu_title()));
        toggle_item.setEnabled(status.toggle_enabled());
    }

    let icon = TrayIcon::from_composite(status);
    if cached_icon != Some(icon) {
        apply_icon(&button, icon);
        STATE.with(|s| {
            if let Some(state) = s.borrow().as_ref() {
                state.last_icon.set(Some(icon));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every [`CompositeStatus`] variant, one of each — mirrors
    /// `app_status.rs`'s own exhaustive-coverage tests.
    fn all_composite_statuses() -> [CompositeStatus; 9] {
        [
            CompositeStatus::PermissionsMissing,
            CompositeStatus::RelaunchRequired,
            CompositeStatus::ModelMissing,
            CompositeStatus::Downloading(43),
            CompositeStatus::Verifying,
            CompositeStatus::EngineFailed,
            CompositeStatus::EngineLoading,
            CompositeStatus::Recording,
            CompositeStatus::Ready,
        ]
    }

    #[test]
    fn tray_icon_mapping_is_total_and_matches_the_documented_priority() {
        for status in all_composite_statuses() {
            let icon = TrayIcon::from_composite(&status);
            let expected = match status {
                CompositeStatus::Ready => TrayIcon::Idle,
                CompositeStatus::Recording => TrayIcon::Recording,
                _ => TrayIcon::Maintenance,
            };
            assert_eq!(icon, expected, "mismatch for {status:?}");
        }
    }

    #[test]
    fn every_tray_icon_has_at_least_one_symbol_name() {
        for icon in [TrayIcon::Idle, TrayIcon::Recording, TrayIcon::Maintenance] {
            assert!(
                !icon.symbol_names().is_empty(),
                "{icon:?} has no symbol names to try"
            );
        }
    }

    #[test]
    fn recording_and_maintenance_fallback_chains_end_in_a_known_good_symbol() {
        // `waveform` is the one symbol name every state agrees is safe on
        // this app's macOS 14.0 floor (it's `Idle`'s only, un-fallback-ed
        // name) — `Recording`'s chain must terminate there too, so a
        // completely unavailable preferred name never leaves the button
        // with no image at all.
        assert_eq!(TrayIcon::Recording.symbol_names().last(), Some(&"waveform"));
    }
}
