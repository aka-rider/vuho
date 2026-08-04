//! macOS menu-bar status item (ADR-006 / GAP 2, ADR-020 model-status
//! extension, UI-rehaul WP4 click-split/`StatusModel` wiring, WP6 single-
//! panel integration).
//!
//! The app is `LSUIElement` (no Dock icon, accessory activation policy), so the
//! status item is the only always-available way to quit — and, once a
//! dictation session can exist, to toggle it without the `CapsLock` hotkey.
//!
//! [`install`] serves **both** of the app's two startup states with one menu
//! shape — a click-split button (plain left click opens the panel,
//! right-click/control-click pops a menu) with a Start/Stop toggle ·
//! "Open Vuho" · Quit menu, driven by [`sync`] from the caller's
//! `StatusModel` `Entity`:
//! - **Permissions/relaunch blocked** (`main.rs`'s gate path): `install`
//!   is called with `cmd_tx: None` — there is no dictation session yet, so
//!   the toggle item is a no-op (and, via `sync`, disabled — see
//!   `CompositeStatus::toggle_enabled`); "Open Vuho" still opens the panel
//!   on its Settings tab, where the permission rows / relaunch button live.
//! - **Production** (`wiring::wire_production`): `install` is called with
//!   `cmd_tx: Some(..)` — the toggle sends real `DictationCommand`s.
//!
//! Built with typed objc2-app-kit bindings (like `vuho-os-integration`'s
//! `NSPasteboard` usage). `NSStatusItem`/`NSMenu`/`NSMenuItem` are `MainThreadOnly`,
//! so every call threads a [`MainThreadMarker`]; `install` must run on the
//! main thread (it is — always called from inside the GPUI `run` closure).
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
use crate::main_queue;

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
    // F14: clear any previously-set image first — otherwise a stale icon
    // from an earlier, successful `apply_icon` call could keep showing
    // beside the text glyph `setTitle` is about to add.
    button.setImage(None);
    button.setTitle(&NSString::from_str("𝗏"));
}

// ── Delegate ─────────────────────────────────────────────────────────────

/// Instance variables for the status-item delegate. `cmd_tx` is `None`
/// while every dictation command would be meaningless — the
/// permissions/relaunch-blocked startup path (`main.rs`) — in which case
/// `toggle_listening:` is a no-op (the toggle item is also disabled via
/// `sync`'s `CompositeStatus::toggle_enabled`, so this is a defensive
/// second line, not the primary guard).
struct DelegateIvars {
    cmd_tx: Option<Sender<DictationCommand>>,
    ui_tx: Sender<UiCommand>,
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
            if let Some(cmd_tx) = &self.ivars().cmd_tx {
                log::info!("status_bar: menu toggle → Toggle");
                // Best-effort: the bridge/session may already be gone on shutdown.
                let _ = cmd_tx.send(DictationCommand::Toggle);
            }
        }

        #[unsafe(method(openPanel:))]
        fn open_panel(&self, _sender: Option<&AnyObject>) {
            // Best-effort: the GPUI drain task may already be gone on shutdown.
            let _ = self.ivars().ui_tx.send(UiCommand::OpenPanel);
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
                let _ = self.ivars().ui_tx.send(UiCommand::OpenPanel);
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
    fn new(cmd_tx: Option<Sender<DictationCommand>>, ui_tx: Sender<UiCommand>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(DelegateIvars { cmd_tx, ui_tx });
        // SAFETY: `this` is a freshly allocated, ivar-initialized
        // `StatusDelegate` from `alloc().set_ivars(..)` — calling
        // `[super init]` on it exactly once is the standard, required
        // Objective-C two-phase (`alloc` + `init`) construction sequence
        // `define_class!`-generated subclasses must follow.
        unsafe { msg_send![super(this), init] }
    }
}

// ── Shared item/menu construction ──────────────────────────────────────────

/// Retained `AppKit` objects for the status item, held for the app lifetime.
/// `last_icon` starts `None` (nothing applied via [`sync`] yet, distinct
/// from whatever static default [`configure_status_button`] painted at
/// install time) and is only ever read/written from the main thread.
struct StatusState {
    item: Retained<NSStatusItem>,
    button: Retained<NSStatusBarButton>,
    menu: Retained<NSMenu>,
    toggle_item: Retained<NSMenuItem>,
    // Retained for the process lifetime only — `setTarget:` (used by the
    // production click-split) does not itself retain its target, so this is
    // what keeps the delegate alive; never read again after construction.
    _delegate: Retained<StatusDelegate>,
    last_icon: Cell<Option<TrayIcon>>,
}

thread_local! {
    /// Main-thread-only storage for the currently-installed status item —
    /// `main.rs` installs exactly one per process. Populated by [`install`],
    /// read by [`sync`]/[`pop_menu`]/`statusItemClicked:`; all run on the
    /// GPUI main thread.
    static STATE: RefCell<Option<StatusState>> = const { RefCell::new(None) };
}

/// Install the menu-bar status item with a click-split button (plain left
/// click opens the panel, right-click/control-click pops the menu) and a
/// Start/Stop toggle · "Open Vuho" · Quit menu.
///
/// Must be called on the main thread (from the GPUI `run` closure), exactly
/// once per process — both `main.rs` entry paths (permissions/relaunch
/// blocked, and production) call this, never [`install`] a second time.
/// `cmd_tx` is `None` on the blocked path (no dictation session exists yet —
/// the toggle no-ops, see [`DelegateIvars`]) and `Some(..)` in production,
/// where it sends [`DictationCommand::Toggle`] — the same channel the
/// `CapsLock` hotkey uses. Both "Open Vuho" and a plain click send
/// [`UiCommand::OpenPanel`] on `ui_tx`, drained by the GPUI foreground task
/// that owns window creation.
///
/// Paints the tray's first frame itself, from `initial` (F21) — the caller
/// does **not** need to follow this with its own [`sync`] call. This isn't
/// just convenience: the `cx.observe(&status, ..)` observer both `main.rs`
/// call sites register calls [`sync`] on every `StatusModel` change, and
/// that observer is set up *before* `install` ever runs — so any change
/// observed in that window hits [`sync`]'s own `STATE`-not-yet-populated
/// early return and is silently dropped from the tray's perspective. Folding
/// the first paint into `install` itself (reading whatever `initial` is at
/// the moment `install` actually runs, past all of that) is what closes that
/// gap, rather than relying on the caller to duplicate a `sync` call right
/// after — a pairing that was easy to get right at each of two call sites
/// today, but load-bearing in a way a future third call site could silently
/// get wrong.
pub(crate) fn install(
    cmd_tx: Option<Sender<DictationCommand>>,
    ui_tx: Sender<UiCommand>,
    initial: &CompositeStatus,
) {
    let Some(mtm) = MainThreadMarker::new() else {
        log::error!("status_bar::install must be called on the main thread");
        return;
    };

    let delegate = StatusDelegate::new(cmd_tx, ui_tx);
    let bar = NSStatusBar::systemStatusBar();
    let item = bar.statusItemWithLength(NSVariableStatusItemLength);
    let Some(button) = configure_status_button(&item, mtm) else {
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
            toggle_item,
            _delegate: delegate,
            last_icon: Cell::new(None),
        });
    });

    sync(initial);
}

/// Pop the production menu via the modern status-item trick: attach it,
/// simulate a click to run `AppKit`'s own menu-tracking loop, then detach —
/// so the button can stay click-split (a plain click reaches
/// `statusItemClicked:` instead of a permanently attached menu intercepting
/// every click first).
///
/// The whole `setMenu(Some)`/`performClick:`/`setMenu(None)` sequence runs
/// inside a single [`main_queue::defer`]red closure (CONSTITUTION rule 33):
/// `performClick:` pumps a nested run loop, so it must never run from inside
/// a live gpui `App` borrow — deferring lets `statusItemClicked:`'s own
/// call stack unwind first, so the nested loop starts from a guaranteed-clean
/// stack rather than from whichever stack happened to call this. Re-derives
/// the `Retained` handles it needs from [`STATE`] *inside* the deferred
/// closure (this module's borrow-discipline rule — see the module doc
/// comment) rather than capturing them, since `Retained<T>` isn't `Send`.
fn pop_menu() {
    main_queue::defer(|| {
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
    });
}

/// Set the status-item button's initial icon — the waveform icon is how the
/// user recognizes Vuho is running at all, before `install`'s caller ever
/// calls [`sync`]. Returns the button so callers can build the rest of
/// [`StatusState`] around it.
///
/// Goes through [`apply_icon`]'s own `TrayIcon::Idle` fallback chain (F9)
/// rather than hardcoding a second, independent "waveform" symbol lookup +
/// "𝗏" text fallback here — one source of truth for how a symbol name
/// resolves to an actual icon, not two that could drift apart.
fn configure_status_button(
    item: &NSStatusItem,
    mtm: MainThreadMarker,
) -> Option<Retained<NSStatusBarButton>> {
    let button = item.button(mtm)?;
    apply_icon(&button, TrayIcon::Idle);
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

/// The **only** tray mutator: reflect one [`CompositeStatus`] in the toggle
/// item's title/enabled state and the button's icon (via [`TrayIcon`]).
///
/// Called from the GPUI foreground task (main thread) — same thread as
/// [`install`], so it sees the populated `thread_local`. A no-op before any
/// install has run.
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

    toggle_item.setTitle(&NSString::from_str(&status.menu_title()));
    toggle_item.setEnabled(status.toggle_enabled());

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
        // F13: `Maintenance`'s chain doesn't end in `waveform` — it ends in
        // `exclamationmark.triangle`, documented (`TrayIcon::symbol_names`)
        // as 12.0+ and thus safe on this app's 14.0 floor too. Pinning its
        // exact name here catches an accidental reorder/typo the same way
        // the `Recording` assertion above does.
        assert_eq!(
            TrayIcon::Maintenance.symbol_names().last(),
            Some(&"exclamationmark.triangle")
        );
    }
}
