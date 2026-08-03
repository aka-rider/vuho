//! Build script for vuho-stt-engine.
//!
//! Embeds `packaging/Info.plist` (`NSMicrophoneUsageDescription`) into this
//! crate's test binaries via `-sectcreate __TEXT __info_plist`. Without it,
//! `#[ignore]`d tests that touch the microphone (e.g. `streaming_smoke` in
//! `tests/streaming.rs`) get silently denied rather than prompted, same as a
//! bare `cargo run -p vuho-ui` would without `crates/vuho-ui/build.rs`.
//!
//! On non-macOS targets, degrades gracefully with a warning.

use std::path::PathBuf;

fn main() {
    // Guard: only build on macOS.
    if !cfg_target_macos() {
        println!("cargo:warning=macOS-only build step skipped on this target");
        return;
    }

    embed_info_plist_in_test_binaries();
}

/// Embed `packaging/Info.plist` (`NSMicrophoneUsageDescription`) into this
/// crate's test binaries via `-sectcreate __TEXT __info_plist`. Without it,
/// `#[ignore]`d tests that touch the microphone (e.g. `streaming_smoke` in
/// `tests/streaming.rs`) get silently denied rather than prompted, same as a
/// bare `cargo run -p vuho-ui` would without `crates/vuho-ui/build.rs`.
fn embed_info_plist_in_test_binaries() {
    let plist = find_workspace_root().join("packaging").join("Info.plist");
    println!("cargo:rerun-if-changed={}", plist.display());
    if !plist.exists() {
        println!(
            "cargo:warning=packaging/Info.plist not found at {}; test binaries will have no embedded Info.plist",
            plist.display()
        );
        return;
    }
    println!("cargo:rustc-link-arg-tests=-sectcreate");
    println!("cargo:rustc-link-arg-tests=__TEXT");
    println!("cargo:rustc-link-arg-tests=__info_plist");
    println!("cargo:rustc-link-arg-tests={}", plist.display());
}

/// Find the workspace root by walking up from `CARGO_MANIFEST_DIR` looking
/// for the top-level `Cargo.toml` with a `[workspace]` table.
fn find_workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..10 {
        let candidate = dir.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(contents) = std::fs::read_to_string(&candidate) {
                if contents.contains("[workspace]") {
                    return dir;
                }
            }
        }
        if !dir.pop() {
            break;
        }
    }
    // Fallback: two levels up from crates/vuho-stt-engine.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map_or_else(|| PathBuf::from("."), std::path::Path::to_path_buf)
}

/// Check if we're targeting macOS.
fn cfg_target_macos() -> bool {
    std::env::var("TARGET")
        .unwrap_or_default()
        .contains("darwin")
}
