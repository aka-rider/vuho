//! User-configurable settings: persistence + defaults.
//!
//! Stores the global hotkey preset and the selected microphone device in
//! `$XDG_CONFIG_HOME/vuho/settings.json` (falling back to
//! `$HOME/.config/vuho/settings.json`), written atomically (temp file +
//! rename in the same directory).
//!
//! Deliberately serde-only: this crate knows nothing about `CGEventFlags`,
//! `HotkeyConfig`, or any platform type — the mapping from [`HotkeySetting`]
//! to a concrete hotkey configuration lives in `vuho-ui` (the composition
//! root that already depends on both `vuho-settings` and
//! `vuho-os-integration`).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

/// Settings file name, relative to the config directory.
const SETTINGS_FILENAME: &str = "settings.json";
/// Vuho's config subdirectory, relative to the XDG/HOME config root.
const CONFIG_SUBDIR: &str = "vuho";

/// The `Settings` schema version this build writes and understands.
///
/// Bump this and extend [`Settings::migrate`] whenever the schema changes
/// in a way that needs translation from an older on-disk shape.
const CURRENT_SETTINGS_VERSION: u32 = 1;

/// A named global-hotkey preset.
///
/// Closed enum (no key-capture recorder — see the settings-window plan):
/// the settings UI offers a fixed dropdown of presets. Serialized
/// `snake_case` so the on-disk JSON is human-readable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HotkeySetting {
    /// `CapsLock` tap-to-toggle (ADR-007 default).
    #[default]
    CapsLock,
    /// Option + Space chord.
    OptionSpace,
    /// Control + Option + Space chord.
    ControlOptionSpace,
    /// Command + Shift + Space chord.
    CommandShiftSpace,
    /// Control + Option + D chord.
    ControlOptionD,
}

impl HotkeySetting {
    /// Every preset, in the order the settings dropdown lists them.
    pub const ALL: [Self; 5] = [
        Self::CapsLock,
        Self::OptionSpace,
        Self::ControlOptionSpace,
        Self::CommandShiftSpace,
        Self::ControlOptionD,
    ];

    /// Human-readable label for the settings dropdown.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::CapsLock => "CapsLock",
            Self::OptionSpace => "⌥ Space",
            Self::ControlOptionSpace => "⌃⌥ Space",
            Self::CommandShiftSpace => "⌘⇧ Space",
            Self::ControlOptionD => "⌃⌥ D",
        }
    }
}

/// All user-configurable settings, persisted as a single JSON document.
///
/// `version` is checked by `Settings::migrate` (crate-private) before this
/// struct's own `Deserialize` ever runs, but `migrate`'s pre-check does not
/// exempt the struct-level deserialization from also needing a default: a
/// pre-versioning file (every real-world version-1 file — the field was
/// added after v1 already shipped) has no `"version"` key at all, and
/// without `#[serde(default = ...)]` here `serde_json::from_value` would
/// reject the whole document as missing a required field, discarding the
/// user's other settings. A file with no `version` key **is** version 1 by
/// definition, so the default matches `CURRENT_SETTINGS_VERSION`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    /// Schema version this document was written at. See `Settings::migrate`
    /// (crate-private).
    #[serde(default = "default_settings_version")]
    pub version: u32,
    /// The global hotkey preset. Missing in an older settings file →
    /// [`HotkeySetting::default`].
    #[serde(default)]
    pub hotkey: HotkeySetting,
    /// The configured input device name, or `None` for the system default.
    /// `AudioDeviceID`s are unstable across reboots, so the name (not the
    /// numeric ID) is what's persisted; `vuho-audio` resolves it back to a
    /// device by name at stream start (see `vuho_audio::start_capture`),
    /// falling back to the system default when the name is absent or no
    /// longer resolves (e.g. the device was unplugged).
    #[serde(default)]
    pub microphone: Option<String>,
}

/// `#[serde(default = ...)]` target for [`Settings::version`]: a document
/// with no `"version"` key predates the field's introduction and is version
/// 1 by definition — the only version that has ever shipped without it.
fn default_settings_version() -> u32 {
    CURRENT_SETTINGS_VERSION
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: CURRENT_SETTINGS_VERSION,
            hotkey: HotkeySetting::default(),
            microphone: None,
        }
    }
}

impl Settings {
    /// Interpret a raw parsed JSON document as `Settings`, handling schema
    /// version compatibility — the single chokepoint every load path routes
    /// through (`SettingsStore::load_from`).
    ///
    /// - A `version` field matching [`CURRENT_SETTINGS_VERSION`] (or a file
    ///   with no `version` field at all, i.e. one written before versioning
    ///   existed — treated as version 1, the only version that has ever
    ///   shipped) deserializes normally, tolerating missing/extra fields via
    ///   `#[serde(default)]`.
    /// - Any other `version` is one this build doesn't understand: logs a
    ///   warning and returns `Settings::default()` **without** touching the
    ///   file — extending the existing malformed-file policy (never
    ///   overwrite a file this build didn't fully understand) to version
    ///   skew, e.g. a newer build having written a schema an older build
    ///   doesn't know how to migrate.
    /// - A `version` field present but structurally malformed elsewhere
    ///   (fields with the wrong type) falls back to `Settings::default()`
    ///   with a warning, same as today's malformed-JSON handling.
    ///
    /// Returns the resolved `Settings` alongside a human-readable reason
    /// whenever it fell back to defaults (`None` on a clean version-matched
    /// load) — [`SettingsStore::load_from`] folds this together with its own
    /// "file isn't valid JSON at all" case into [`SettingsStore::load_warning`],
    /// the one place a caller (e.g. a Settings-tab notice banner) can learn
    /// that defaults are in use without re-parsing the file itself.
    fn migrate(raw: &serde_json::Value) -> (Settings, Option<String>) {
        let version = raw
            .get("version")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(u64::from(CURRENT_SETTINGS_VERSION));

        if version != u64::from(CURRENT_SETTINGS_VERSION) {
            let reason = format!(
                "settings file has version {version}, this build only understands version \
                 {CURRENT_SETTINGS_VERSION} — using defaults; file left untouched"
            );
            log::warn!("{reason}");
            return (Settings::default(), Some(reason));
        }

        match serde_json::from_value(raw.clone()) {
            Ok(settings) => (settings, None),
            Err(e) => {
                let reason = format!(
                    "settings file is malformed ({e}) — using defaults; file left untouched"
                );
                log::warn!("{reason}");
                (Settings::default(), Some(reason))
            }
        }
    }
}

/// Errors from loading or saving the settings file.
#[derive(thiserror::Error, Debug)]
pub enum SettingsError {
    /// The settings file (or its temp file / parent directory) could not be
    /// read or written.
    #[error("settings I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The settings file's JSON could not be serialized.
    #[error("settings serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Owns the on-disk settings file and an in-memory cache.
///
/// One instance should exist per process (created in `main`, injected via
/// `Arc` into every consumer) — it is the sole owner of the settings file.
pub struct SettingsStore {
    path: PathBuf,
    current: RwLock<Settings>,
    /// Human-readable reason the load fell back to defaults, captured once
    /// at [`Self::load_from`] time — `None` on a clean load. Immutable for
    /// the store's lifetime: a load-time fact, not something `update()`
    /// changes. See [`Self::load_warning`].
    load_warning: Option<String>,
}

impl SettingsStore {
    /// The default settings file path: `$XDG_CONFIG_HOME/vuho/settings.json`
    /// if `XDG_CONFIG_HOME` is set and non-empty, else
    /// `$HOME/.config/vuho/settings.json`.
    #[must_use]
    pub fn default_path() -> PathBuf {
        let config_root = std::env::var("XDG_CONFIG_HOME")
            .ok()
            .filter(|v| !v.is_empty())
            .map_or_else(
                || {
                    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                    Path::new(&home).join(".config")
                },
                PathBuf::from,
            );
        config_root.join(CONFIG_SUBDIR).join(SETTINGS_FILENAME)
    }

    /// Load settings from [`Self::default_path`], or fall back to defaults.
    ///
    /// Never fails: a missing file yields defaults (no write); a malformed
    /// file logs a warning and yields defaults, without touching the file
    /// on disk (the user's changes, if the file is later fixed by hand,
    /// are only overwritten once they change a setting through the UI).
    #[must_use]
    pub fn load_or_default() -> Self {
        Self::load_from(Self::default_path())
    }

    /// Load settings from an explicit path (used directly by tests; the
    /// production entry point is [`Self::load_or_default`]).
    #[must_use]
    pub fn load_from(path: PathBuf) -> Self {
        let (settings, load_warning) = match fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str::<serde_json::Value>(&contents) {
                Ok(raw) => Settings::migrate(&raw),
                Err(e) => {
                    let reason = format!(
                        "settings file at {} is malformed ({e}) — using defaults; file left untouched",
                        path.display()
                    );
                    log::warn!("{reason}");
                    (Settings::default(), Some(reason))
                }
            },
            // A missing file is the expected, unremarkable first-run case —
            // not a warning-worthy fallback.
            Err(_) => (Settings::default(), None),
        };
        Self {
            path,
            current: RwLock::new(settings),
            load_warning,
        }
    }

    /// A human-readable reason the load fell back to defaults — a malformed
    /// file, or a `version` this build doesn't understand — or `None` on a
    /// clean load (including "no file existed yet"). Captured once at load
    /// time; not re-checked on every [`Self::get`]/[`Self::update`] call.
    #[must_use]
    pub fn load_warning(&self) -> Option<&str> {
        self.load_warning.as_deref()
    }

    /// Return a clone of the current in-memory settings.
    #[must_use]
    pub fn get(&self) -> Settings {
        match self.current.read() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => {
                log::error!("settings lock poisoned, recovering");
                poisoned.into_inner().clone()
            }
        }
    }

    /// Mutate the in-memory settings via `f`, then persist atomically
    /// (write to a temp file in the same directory, then `rename`).
    ///
    /// The write lock is held across both the mutation and the disk write,
    /// so two racing `update()` calls are fully serialized rather than
    /// merely serializing their in-memory mutation: releasing the lock
    /// between "mutate" and "persist" let two calls' disk writes race
    /// independently of the order their mutations actually landed in
    /// memory, so the file on disk could end up holding an *older* update's
    /// value than what's in memory. Holding the lock across `write_to_disk`
    /// makes each `update()` an atomic mutate-and-persist unit, so disk
    /// order always matches mutation order.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsError::Io`] if the parent directory can't be
    /// created or the temp file can't be written/renamed, or
    /// [`SettingsError::Json`] if serialization fails.
    pub fn update(&self, f: impl FnOnce(&mut Settings)) -> Result<(), SettingsError> {
        let mut guard = match self.current.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                log::error!("settings lock poisoned, recovering");
                poisoned.into_inner()
            }
        };
        f(&mut guard);
        self.write_to_disk(&guard)
    }

    /// Serialize `settings` and atomically replace the settings file.
    ///
    /// The temp-file + `sync_all` + `rename` + best-effort parent-dir-fsync
    /// durability chain itself lives in
    /// [`vuho_model_paths::atomic_write`] — the one copy shared with the
    /// model downloader's sidecar manifest (CONSTITUTION rule 26). This
    /// method's own job is just: ensure the parent directory exists (the
    /// shared helper doesn't create directories, since its callers each
    /// have their own directory-creation step with their own error
    /// handling) and serialize `settings` to pretty JSON first.
    ///
    /// Note: unlike the logic this replaced, a failed best-effort
    /// parent-directory fsync is no longer logged here — `vuho-model-paths`
    /// is deliberately std-only (ADR-019) and carries no `log` dependency.
    /// The failure is still non-fatal and still swallowed, just silently;
    /// see `atomic_write`'s doc comment.
    fn write_to_disk(&self, settings: &Settings) -> Result<(), SettingsError> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut json = serde_json::to_string_pretty(settings)?;
        json.push('\n');

        vuho_model_paths::atomic_write(&self.path, json.as_bytes())?;
        Ok(())
    }

    /// A `SettingsStore` backed by a unique, process-local temp path —
    /// never touches the real `~/.config/vuho/settings.json` and never
    /// collides with a concurrently running test.
    ///
    /// `label` should identify the caller (e.g. `"pipeline"`) to keep temp
    /// paths readable when debugging a leftover file. Test-only: gated by
    /// `#[cfg(test)]` within this crate and by the `test-util` feature for
    /// other crates' test code (`vuho-dictation`'s `test_settings` helpers),
    /// so this is the single source of the "temp settings path" construction
    /// (CONSTITUTION rule 26) instead of being copy-pasted per crate.
    #[cfg(any(test, feature = "test-util"))]
    #[must_use]
    pub fn new_temp(label: &str) -> Self {
        Self::load_from(temp_settings_path(label))
    }
}

/// Build a unique settings-file path under the OS temp directory.
///
/// Uniqueness comes from the process ID plus a process-local monotonic
/// counter (not `Instant`'s opaque `Debug` output, which is not guaranteed
/// unique or even useful for uniqueness across calls).
#[cfg(any(test, feature = "test-util"))]
fn temp_settings_path(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "vuho-settings-test-{label}-{}-{seq}",
        std::process::id()
    ));
    dir.join("settings.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_hotkey_preset_round_trips() {
        for preset in HotkeySetting::ALL {
            let json = serde_json::to_string(&preset).unwrap();
            let back: HotkeySetting = serde_json::from_str(&json).unwrap();
            assert_eq!(back, preset);
        }
    }

    #[test]
    fn hotkey_presets_serialize_snake_case() {
        assert_eq!(
            serde_json::to_string(&HotkeySetting::CapsLock).unwrap(),
            "\"caps_lock\""
        );
        assert_eq!(
            serde_json::to_string(&HotkeySetting::OptionSpace).unwrap(),
            "\"option_space\""
        );
        assert_eq!(
            serde_json::to_string(&HotkeySetting::ControlOptionSpace).unwrap(),
            "\"control_option_space\""
        );
        assert_eq!(
            serde_json::to_string(&HotkeySetting::CommandShiftSpace).unwrap(),
            "\"command_shift_space\""
        );
        assert_eq!(
            serde_json::to_string(&HotkeySetting::ControlOptionD).unwrap(),
            "\"control_option_d\""
        );
    }

    #[test]
    fn missing_file_yields_defaults_and_does_not_write() {
        let path = temp_settings_path("missing");
        let store = SettingsStore::load_from(path.clone());
        assert_eq!(store.get(), Settings::default());
        assert!(!path.exists());
        // A missing file is the ordinary first-run case, not a fallback
        // worth warning the user about.
        assert_eq!(store.load_warning(), None);
    }

    #[test]
    fn malformed_file_yields_defaults_and_leaves_file_untouched() {
        let path = temp_settings_path("malformed");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"not valid json").unwrap();

        let store = SettingsStore::load_from(path.clone());
        assert_eq!(store.get(), Settings::default());

        // The malformed bytes must not have been overwritten by the load.
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes, b"not valid json");

        let warning = store.load_warning().expect("malformed load must warn");
        assert!(warning.contains("malformed"), "warning: {warning:?}");
    }

    #[test]
    fn clean_load_has_no_warning() {
        let path = temp_settings_path("clean-load-warning");
        let store = SettingsStore::load_from(path.clone());
        store
            .update(|s| s.hotkey = HotkeySetting::OptionSpace)
            .unwrap();
        assert_eq!(store.load_warning(), None);

        // A fresh store loading that same, well-formed, current-version file
        // must also see no warning.
        let reloaded = SettingsStore::load_from(path);
        assert_eq!(reloaded.load_warning(), None);
    }

    #[test]
    fn missing_fields_are_tolerated_via_serde_default() {
        let path = temp_settings_path("partial");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"{}").unwrap();

        let store = SettingsStore::load_from(path);
        assert_eq!(store.get(), Settings::default());
    }

    #[test]
    fn update_persists_and_is_reloadable() {
        let path = temp_settings_path("update");
        let store = SettingsStore::load_from(path.clone());

        store
            .update(|s| {
                s.hotkey = HotkeySetting::OptionSpace;
                s.microphone = Some("Test Mic".to_string());
            })
            .unwrap();

        assert_eq!(store.get().hotkey, HotkeySetting::OptionSpace);
        assert_eq!(store.get().microphone.as_deref(), Some("Test Mic"));

        // Re-load from disk into a fresh store — the write must be visible.
        let reloaded = SettingsStore::load_from(path);
        let settings = reloaded.get();
        assert_eq!(settings.hotkey, HotkeySetting::OptionSpace);
        assert_eq!(settings.microphone.as_deref(), Some("Test Mic"));
    }

    #[test]
    fn update_writes_valid_pretty_json_with_trailing_newline() {
        let path = temp_settings_path("json-shape");
        let store = SettingsStore::load_from(path.clone());
        store
            .update(|s| s.hotkey = HotkeySetting::ControlOptionD)
            .unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.ends_with('\n'));
        assert!(contents.contains("control_option_d"));
        // Pretty-printed: multi-line.
        assert!(contents.lines().count() > 1);
    }

    #[test]
    fn default_path_uses_xdg_config_home_when_set() {
        // This test only checks the join logic in isolation from the
        // process environment (which we must not mutate from a parallel
        // test run) by re-deriving the expected suffix directly.
        let expected_suffix = Path::new("vuho").join("settings.json");
        let path = SettingsStore::default_path();
        assert!(path.ends_with(&expected_suffix));
    }

    #[test]
    fn version_roundtrips() {
        let path = temp_settings_path("version-roundtrip");
        let store = SettingsStore::load_from(path.clone());
        assert_eq!(store.get().version, CURRENT_SETTINGS_VERSION);

        store
            .update(|s| s.hotkey = HotkeySetting::OptionSpace)
            .unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        let raw: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(
            raw.get("version").and_then(serde_json::Value::as_u64),
            Some(u64::from(CURRENT_SETTINGS_VERSION))
        );

        // A fresh load must see the same version.
        let reloaded = SettingsStore::load_from(path);
        assert_eq!(reloaded.get().version, CURRENT_SETTINGS_VERSION);
    }

    #[test]
    fn future_version_preserves_file_bytes() {
        let path = temp_settings_path("future-version");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let original = br#"{"version":999,"garbage_field":"garbage_value","hotkey":"caps_lock"}"#;
        fs::write(&path, original).unwrap();

        let store = SettingsStore::load_from(path.clone());

        // Falsification target: an unrecognized future version must warn +
        // fall back to defaults, exactly like the malformed-JSON path —
        // never partially adopt fields from a schema this build doesn't
        // understand.
        assert_eq!(store.get(), Settings::default());

        // The file on disk must be byte-identical to what was there before
        // load — loading never writes. Comparing the full byte vector is a
        // strictly stronger check than a checksum (no collision possibility)
        // while still directly answering "did the load touch the file?".
        let bytes_after = fs::read(&path).unwrap();
        assert_eq!(
            bytes_after, original,
            "loading a future-version settings file must not modify it on disk"
        );

        let warning = store
            .load_warning()
            .expect("an unrecognized future version must populate load_warning");
        assert!(warning.contains("999"), "warning: {warning:?}");
    }

    #[test]
    fn pre_versioning_file_without_version_field_preserves_values() {
        // A file written before `version` existed (i.e. every real-world v1
        // file) has no `"version"` key at all. `migrate` treats a missing
        // key as version 1 (the only version that has ever shipped) — but
        // that value must actually survive the subsequent `Settings`
        // deserialization, not just the version-number check. Falsification
        // target: without `#[serde(default = ...)]` on `Settings::version`,
        // serde's own struct deserialization rejects the file for the
        // missing `version` field, discarding the user's hotkey/microphone.
        let path = temp_settings_path("no-version-field");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            br#"{"hotkey":"option_space","microphone":"Test Mic"}"#,
        )
        .unwrap();

        let store = SettingsStore::load_from(path);
        let settings = store.get();
        assert_eq!(settings.hotkey, HotkeySetting::OptionSpace);
        assert_eq!(settings.microphone.as_deref(), Some("Test Mic"));
        assert_eq!(settings.version, CURRENT_SETTINGS_VERSION);
    }

    #[test]
    fn concurrent_update_no_temp_clobber() {
        use std::sync::Arc;
        use std::thread;

        let path = temp_settings_path("concurrent-update");
        let store = Arc::new(SettingsStore::load_from(path.clone()));

        // Two threads racing `update()` on the same store must each get a
        // unique temp file (per-call PID+counter in `tmp_path`) — neither
        // write's temp file may collide with the other's before its own
        // `rename` lands, and both `update()` calls must return `Ok`.
        let store_a = Arc::clone(&store);
        let handle_a = thread::spawn(move || {
            for _ in 0..20 {
                store_a
                    .update(|s| s.hotkey = HotkeySetting::OptionSpace)
                    .unwrap();
            }
        });
        let store_b = Arc::clone(&store);
        let handle_b = thread::spawn(move || {
            for _ in 0..20 {
                store_b
                    .update(|s| s.hotkey = HotkeySetting::ControlOptionD)
                    .unwrap();
            }
        });

        handle_a.join().unwrap();
        handle_b.join().unwrap();

        // The file must exist and be valid, current-version JSON afterward —
        // whichever thread's write landed last, nothing corrupted it.
        let contents = fs::read_to_string(&path).unwrap();
        let final_settings: Settings = serde_json::from_str(&contents).unwrap();
        assert_eq!(final_settings.version, CURRENT_SETTINGS_VERSION);
        assert!(matches!(
            final_settings.hotkey,
            HotkeySetting::OptionSpace | HotkeySetting::ControlOptionD
        ));

        // Falsification target for the "write outside the lock" bug: the
        // in-memory state (whichever update() call's mutation actually
        // landed last) must match what's on disk exactly. If the write
        // lock were released before write_to_disk (as it used to be), the
        // two threads' disk writes could race independently of their
        // mutation order and leave disk holding a stale value relative to
        // memory — this equality would then intermittently fail.
        assert_eq!(store.get(), final_settings);
    }
}
