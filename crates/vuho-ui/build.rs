//! Build script for vuho-ui.
//!
//! Embeds `packaging/Info.plist` into the `vuho` binary itself (via
//! `-sectcreate __TEXT __info_plist`), independent of whether the binary is
//! later packaged into `Vuho.app`. Without an embedded plist, macOS silently
//! denies microphone access instead of prompting — a bare `cargo run` build
//! has no `Info.plist` otherwise. `scripts/bundle-macos.sh` also copies the
//! same file into `Contents/Info.plist` for Launch Services; both read from
//! this single source of truth, so there is no drift.
//!
//! Link args are emitted (not `.cargo/config.toml` rustflags) because
//! rustflags in `[target.<triple>]` apply to *every* artifact built for that
//! target, including build scripts and proc-macros compiled from crates.io
//! registry checkouts whose CWD is not the workspace root — a relative
//! `packaging/Info.plist` path fails to resolve there. `cargo:rustc-link-arg-bin`
//! only affects this crate's `vuho` binary, and the path is computed from
//! `CARGO_MANIFEST_DIR`, which is always absolute.

use std::path::PathBuf;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let plist = manifest_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // workspace root
        .expect("vuho-ui manifest dir has no workspace-root ancestor")
        .join("packaging")
        .join("Info.plist");

    println!("cargo:rerun-if-changed={}", plist.display());

    if !plist.exists() {
        println!(
            "cargo:warning=packaging/Info.plist not found at {}; the built binary will have no embedded Info.plist and macOS will silently deny microphone access instead of prompting",
            plist.display()
        );
        return;
    }

    println!("cargo:rustc-link-arg-bin=vuho=-sectcreate");
    println!("cargo:rustc-link-arg-bin=vuho=__TEXT");
    println!("cargo:rustc-link-arg-bin=vuho=__info_plist");
    println!("cargo:rustc-link-arg-bin=vuho={}", plist.display());
}
