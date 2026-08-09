//! Keyboard input language detection via macOS Text Input Services (TIS).
//!
//! Maps the current OS keyboard language (BCP-47 tag from TIS) to Whisper-style
//! language codes. Unmapped languages fall back to engine auto-detect (ADR-009).
//!
//! # TIS is main-thread-only
//!
//! macOS asserts (`dispatch_assert_queue` inside `TSMGetInputSourceProperty…`)
//! that Text Input Services is used from the main thread — calling it from a
//! background thread is an uncatchable `SIGTRAP`, not an error. This module
//! makes that misuse unrepresentable: [`LanguageDetector::current_input_language`]
//! requires a [`MainThreadMarker`], and background consumers (the dictation
//! pipeline) read [`cached_input_language`] instead — a cache owned and
//! refreshed exclusively on the main thread by [`install_language_watcher`]
//! (initial read at startup + re-read on every input-source change
//! notification).

use std::sync::{OnceLock, RwLock};

use objc2::MainThreadMarker;

use crate::sys;

/// Last keyboard language observed on the main thread. `None` until
/// [`install_language_watcher`] runs, or when the language is unmapped —
/// both mean "let the engine auto-detect" downstream.
static CACHED_LANGUAGE: RwLock<Option<String>> = RwLock::new(None);

/// Every BCP-47 primary subtag this module maps, and — since every mapping
/// is the identity — every Whisper code it can produce. The single source
/// both [`map_bcp47_to_whisper`] and [`mapped_languages`] read, so the
/// supported set cannot drift from the mapper (CONSTITUTION rule 26).
const MAPPED_LANGUAGES: [&str; 29] = [
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "ja", "ko", "lt",
    "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk", "vi", "zh",
];

/// Map a BCP-47 language tag to a Whisper language code.
///
/// Extracts the primary subtag (everything before the first `-`, e.g. `"en"`
/// from `"en-US"`) and looks *only that* up in [`MAPPED_LANGUAGES`] — the
/// primary subtag by construction never contains a `-`, so full tags like
/// `"en-US"` are never themselves listed. Returns `None` for unmapped
/// languages, which signals the caller to use engine auto-detect (ADR-009).
///
/// See [`mapped_languages`] for the supported set.
#[must_use]
pub fn map_bcp47_to_whisper(tag: &str) -> Option<&'static str> {
    let primary = tag.split('-').next()?;
    MAPPED_LANGUAGES.into_iter().find(|&code| code == primary)
}

/// Every Whisper language code [`map_bcp47_to_whisper`] can return.
///
/// Callers intersect this with a speech model's own supported set to tell
/// the user which languages that model can actually be reached in.
#[must_use]
pub fn mapped_languages() -> &'static [&'static str] {
    &MAPPED_LANGUAGES
}

/// Detects the current keyboard input language via macOS TIS.
pub struct LanguageDetector;

impl LanguageDetector {
    /// Returns the current keyboard input language as a Whisper language code.
    ///
    /// Queries TIS for the current input source's primary language, then maps
    /// the BCP-47 tag to a Whisper code via [`map_bcp47_to_whisper`].
    ///
    /// # Spec
    ///
    /// "STT language always matches OS-native keyboard input method."
    ///
    /// If the detected language is unmapped or TIS is unavailable
    /// (CI, headless), returns `Err(OsError::LanguageDetection)` so the
    /// caller can fall back to engine auto-detect (ADR-009).
    ///
    /// # Errors
    ///
    /// Returns `OsError::LanguageDetection` if the input source cannot be
    /// queried or the language is unmapped.
    pub fn current_input_language(_mtm: MainThreadMarker) -> Result<String, crate::OsError> {
        // The marker is the whole point: TIS traps (uncatchable SIGTRAP via
        // dispatch_assert_queue) when called off the main thread.
        let tag = sys::tis_current_language().ok_or(crate::OsError::LanguageDetection)?;
        let whisper_code = map_bcp47_to_whisper(&tag).ok_or(crate::OsError::LanguageDetection)?;
        Ok(whisper_code.to_string())
    }
}

/// Read the last keyboard language observed by the main-thread watcher.
///
/// Safe from any thread. Returns `None` until [`install_language_watcher`]
/// has run (or when the current language is unmapped) — callers treat that
/// as "engine auto-detect" (ADR-009).
#[must_use]
pub fn cached_input_language() -> Option<String> {
    CACHED_LANGUAGE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Re-read TIS (main thread) and update [`CACHED_LANGUAGE`].
fn refresh_cached_language(mtm: MainThreadMarker) {
    let lang = LanguageDetector::current_input_language(mtm).ok();
    log::info!("language watcher: keyboard language = {lang:?}");
    *CACHED_LANGUAGE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = lang;
}

/// The retained observer object returned by `addObserverForName:…`. Held
/// only so the observation lives as long as the app (stored in
/// [`OBSERVER`]) — never read again after construction.
struct ObserverToken(#[allow(dead_code)] objc2::rc::Retained<objc2::runtime::AnyObject>);
// SAFETY: the token is write-once, never read, never sent across threads
// for use — it exists purely to pin the observer registration alive.
unsafe impl Send for ObserverToken {}
unsafe impl Sync for ObserverToken {}

/// App-scoped observer-token storage for [`install_language_watcher`] — see
/// that function's doc for why this is a static (CONSTITUTION rule 3).
static OBSERVER: OnceLock<ObserverToken> = OnceLock::new();

/// Register the `NSDistributedNotificationCenter` observer for TIS's
/// `kTISNotifySelectedKeyboardInputSourceChanged` notification, re-reading
/// and caching the language on every fire. Returns the retained token the
/// caller must keep alive for the app's lifetime.
///
/// Split out of [`install_language_watcher`] purely to keep that function
/// under the line budget (CONSTITUTION rule 28): this half owns the actual
/// objc2 block/observer construction; the other half owns the app-scoped,
/// install-once-only policy around it.
fn register_input_source_observer(mtm: MainThreadMarker) -> ObserverToken {
    use block2::RcBlock;
    use objc2_foundation::{
        NSDistributedNotificationCenter, NSNotification, NSOperationQueue, NSString,
    };

    // TIS posts this on input-source switches (the constant
    // kTISNotifySelectedKeyboardInputSourceChanged).
    let name = NSString::from_str("com.apple.Carbon.TISNotifySelectedKeyboardInputSourceChanged");
    let block = RcBlock::new(move |_notif: std::ptr::NonNull<NSNotification>| {
        // Delivered on the main queue (passed below) — the marker is
        // genuinely available here, not assumed.
        if let Some(mtm) = MainThreadMarker::new() {
            refresh_cached_language(mtm);
        }
    });
    let center = NSDistributedNotificationCenter::defaultCenter();
    let _ = mtm; // marker consumed by the initial refresh; queue getter needs none
    let queue = NSOperationQueue::mainQueue();
    // SAFETY: `block` is retained for the duration of this call and handed
    // to AppKit, which itself retains it as part of registering the
    // observer — the block does not need to outlive this call on its own.
    let token = unsafe {
        center.addObserverForName_object_queue_usingBlock(Some(&name), None, Some(&queue), &block)
    };
    ObserverToken(token.into())
}

/// Install the app-scoped keyboard-language watcher. Main thread only.
///
/// Reads the current language immediately, then re-reads on every
/// `kTISNotifySelectedKeyboardInputSourceChanged` distributed notification
/// (delivered on the main queue, satisfying TIS's main-thread assertion —
/// see `register_input_source_observer` for that half). The observer
/// token lives for the app's lifetime (stored in `OBSERVER` — the
/// watcher is app-scoped by design, like the engine; CONSTITUTION rule 3).
/// Calling more than once is a logged no-op.
pub fn install_language_watcher(mtm: MainThreadMarker) {
    refresh_cached_language(mtm);

    if OBSERVER.set(register_input_source_observer(mtm)).is_err() {
        log::warn!("language watcher: already installed — ignoring duplicate install");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_en_us() {
        assert_eq!(map_bcp47_to_whisper("en-US"), Some("en"));
    }

    #[test]
    fn map_en_gb() {
        assert_eq!(map_bcp47_to_whisper("en-GB"), Some("en"));
    }

    #[test]
    fn map_zh_hans() {
        assert_eq!(map_bcp47_to_whisper("zh-Hans"), Some("zh"));
    }

    #[test]
    fn map_de_at() {
        assert_eq!(map_bcp47_to_whisper("de-AT"), Some("de"));
    }

    #[test]
    fn map_vi_vn() {
        assert_eq!(map_bcp47_to_whisper("vi-VN"), Some("vi"));
    }

    #[test]
    fn map_ja_jp() {
        assert_eq!(map_bcp47_to_whisper("ja-JP"), Some("ja"));
    }

    #[test]
    fn map_ko_kr() {
        assert_eq!(map_bcp47_to_whisper("ko-KR"), Some("ko"));
    }

    #[test]
    fn map_es_es() {
        assert_eq!(map_bcp47_to_whisper("es-ES"), Some("es"));
    }

    #[test]
    fn map_pt_br() {
        assert_eq!(map_bcp47_to_whisper("pt-BR"), Some("pt"));
    }

    #[test]
    fn map_nl_nl() {
        assert_eq!(map_bcp47_to_whisper("nl-NL"), Some("nl"));
    }

    #[test]
    fn map_ru_ru() {
        assert_eq!(map_bcp47_to_whisper("ru-RU"), Some("ru"));
    }

    #[test]
    fn map_it_it() {
        assert_eq!(map_bcp47_to_whisper("it-IT"), Some("it"));
    }

    #[test]
    fn map_fr_fr() {
        assert_eq!(map_bcp47_to_whisper("fr-FR"), Some("fr"));
    }

    #[test]
    fn map_zh_hant() {
        assert_eq!(map_bcp47_to_whisper("zh-Hant"), Some("zh"));
    }

    #[test]
    fn map_bg_bg() {
        assert_eq!(map_bcp47_to_whisper("bg-BG"), Some("bg"));
    }

    #[test]
    fn map_hr_hr() {
        assert_eq!(map_bcp47_to_whisper("hr-HR"), Some("hr"));
    }

    #[test]
    fn map_cs_cz() {
        assert_eq!(map_bcp47_to_whisper("cs-CZ"), Some("cs"));
    }

    #[test]
    fn map_da_dk() {
        assert_eq!(map_bcp47_to_whisper("da-DK"), Some("da"));
    }

    #[test]
    fn map_et_ee() {
        assert_eq!(map_bcp47_to_whisper("et-EE"), Some("et"));
    }

    #[test]
    fn map_fi_fi() {
        assert_eq!(map_bcp47_to_whisper("fi-FI"), Some("fi"));
    }

    #[test]
    fn map_el_gr() {
        assert_eq!(map_bcp47_to_whisper("el-GR"), Some("el"));
    }

    #[test]
    fn map_hu_hu() {
        assert_eq!(map_bcp47_to_whisper("hu-HU"), Some("hu"));
    }

    #[test]
    fn map_lv_lv() {
        assert_eq!(map_bcp47_to_whisper("lv-LV"), Some("lv"));
    }

    #[test]
    fn map_lt_lt() {
        assert_eq!(map_bcp47_to_whisper("lt-LT"), Some("lt"));
    }

    #[test]
    fn map_mt_mt() {
        assert_eq!(map_bcp47_to_whisper("mt-MT"), Some("mt"));
    }

    #[test]
    fn map_pl_pl() {
        assert_eq!(map_bcp47_to_whisper("pl-PL"), Some("pl"));
    }

    #[test]
    fn map_ro_ro() {
        assert_eq!(map_bcp47_to_whisper("ro-RO"), Some("ro"));
    }

    #[test]
    fn map_sk_sk() {
        assert_eq!(map_bcp47_to_whisper("sk-SK"), Some("sk"));
    }

    #[test]
    fn map_sl_si() {
        assert_eq!(map_bcp47_to_whisper("sl-SI"), Some("sl"));
    }

    #[test]
    fn map_sv_se() {
        assert_eq!(map_bcp47_to_whisper("sv-SE"), Some("sv"));
    }

    #[test]
    fn map_uk_ua() {
        assert_eq!(map_bcp47_to_whisper("uk-UA"), Some("uk"));
    }

    #[test]
    fn map_plain_uk() {
        assert_eq!(map_bcp47_to_whisper("uk"), Some("uk"));
    }

    #[test]
    fn every_mapped_language_maps_to_itself() {
        for code in mapped_languages() {
            assert_eq!(map_bcp47_to_whisper(code), Some(*code));
            assert_eq!(map_bcp47_to_whisper(&format!("{code}-XX")), Some(*code));
        }
    }

    #[test]
    fn mapped_languages_are_unique_two_letter_codes() {
        let mut seen = mapped_languages().to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), mapped_languages().len());
        assert!(mapped_languages().iter().all(|code| code.len() == 2));
    }

    #[test]
    fn map_xx_unmapped() {
        assert_eq!(map_bcp47_to_whisper("xx"), None);
    }

    #[test]
    fn map_empty() {
        assert_eq!(map_bcp47_to_whisper(""), None);
    }

    #[test]
    fn map_plain_en() {
        assert_eq!(map_bcp47_to_whisper("en"), Some("en"));
    }

    #[test]
    fn map_plain_de() {
        assert_eq!(map_bcp47_to_whisper("de"), Some("de"));
    }

    #[test]
    fn map_plain_zh() {
        assert_eq!(map_bcp47_to_whisper("zh"), Some("zh"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn detector_returns_valid_code_or_falls_back() {
        let Some(mtm) = objc2::MainThreadMarker::new() else {
            eprintln!("skipping: not on the main thread (TIS is main-thread-only)");
            return;
        };
        if let Ok(lang) = LanguageDetector::current_input_language(mtm) {
            assert_eq!(lang.len(), 2);
        }
    }
}
