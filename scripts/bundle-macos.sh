#!/usr/bin/env bash
# bundle-macos.sh — Assemble Vuho.app for local distribution.
#
# Produces a self-contained Vuho.app with:
#   - The vuho binary in Contents/MacOS/
#   - Parakeet TDT model in Contents/Resources/ (unless VUHO_BUNDLE_MODEL=0)
#   - Valid Info.plist, code-signed with mic entitlement
#
# Two distribution shapes come out of this one script:
#   - VUHO_BUNDLE_MODEL=1 (default) — the ~500 MB DMG-style bundle with the
#     Parakeet-TDT model embedded, offline from first launch.
#   - VUHO_BUNDLE_MODEL=0 — the model-less bundle for Homebrew cask
#     distribution (binary + icon + attribution, no model; measured ≈40 MB
#     on disk / ≈15 MB as the gzipped release tarball — this grows with the
#     binary's own dependencies, so treat it as an order of magnitude, not
#     a promise); the app downloads the model on first run.
#
# Usage:
#   ./scripts/bundle-macos.sh          # builds if stale, ad-hoc signing
#   VUHO_MODEL_DIR=/path/to/model ./scripts/bundle-macos.sh
#   VUHO_BUNDLE_MODEL=0 ./scripts/bundle-macos.sh  # model-less cask bundle
#   SIGN_ID="Vuho Dev" ./scripts/bundle-macos.sh  # stable sig for TCC
#
# Preconditions:
#   - macOS (Apple Silicon)
#   - No separate build step required: this script builds
#     target/release/vuho itself whenever it's missing OR older than the
#     workspace sources (see "Build release artifacts" below) — there is no
#     BUILD=1 flag; nothing ever read one.

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────
#
# The bundle ID, model directory names, and required-component/asset lists
# all come from models.manifest.json (repo root) — the single source of
# truth shared with the Rust build (vuho-model-paths) and the other 3
# scripts, so this bundler can never silently drift from what the app
# itself resolves at runtime.

APP_NAME="Vuho"
SIGN_ID="${SIGN_ID:-}"
VUHO_BUNDLE_MODEL="${VUHO_BUNDLE_MODEL:-1}"

WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$WORKSPACE_ROOT/models.manifest.json"

. "$(dirname "$0")/manifest-lib.sh"

# ── Helpers ─────────────────────────────────────────────────────────────
# (defined before the manifest read below so a malformed manifest can die()
# with a real diagnostic instead of hitting an undefined-function error)

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

manifest_out=$(manifest_vars "$MANIFEST" '
emit("BUNDLE_ID", manifest["bundle_id"])
emit("MODEL_NAME", manifest["stt"]["dir_name"])
emit_array("REQUIRED_COMPONENTS", manifest["stt"]["components"])
') || die "failed to read $MANIFEST (see traceback above)"
eval "$manifest_out"

SRC_BIN="$WORKSPACE_ROOT/target/release/vuho"
SRC_PLIST="$WORKSPACE_ROOT/packaging/Info.plist"
SRC_ENTITLEMENTS="$WORKSPACE_ROOT/packaging/vuho.entitlements"
SRC_ICON="$WORKSPACE_ROOT/packaging/Vuho.icns"
SRC_ATTRIBUTION="$WORKSPACE_ROOT/packaging/ATTRIBUTION.txt"

VUHO_MODEL_DIR="${VUHO_MODEL_DIR:-$WORKSPACE_ROOT/models/$MODEL_NAME}"

APP_DIR="$WORKSPACE_ROOT/$APP_NAME.app"

# ── Precondition checks ────────────────────────────────────────────────

if [[ "$(uname -s)" != "Darwin" ]]; then
    die "This script must run on macOS."
fi

if [[ ! -f "$SRC_PLIST" ]]; then
    die "Info.plist not found at $SRC_PLIST"
fi

if [[ ! -f "$SRC_ENTITLEMENTS" ]]; then
    die "Entitlements not found at $SRC_ENTITLEMENTS"
fi

if [[ ! -f "$SRC_ICON" ]]; then
    die "App icon not found at $SRC_ICON"
fi

if [[ ! -f "$SRC_ATTRIBUTION" ]]; then
    die "ATTRIBUTION.txt not found at $SRC_ATTRIBUTION"
fi

info "Verifying Info.plist's CFBundleIdentifier matches models.manifest.json ($BUNDLE_ID)..."
PLIST_BUNDLE_ID="$(plutil -extract CFBundleIdentifier raw "$SRC_PLIST")"
if [[ "$PLIST_BUNDLE_ID" != "$BUNDLE_ID" ]]; then
    die "Info.plist CFBundleIdentifier ($PLIST_BUNDLE_ID) != models.manifest.json bundle_id ($BUNDLE_ID)"
fi

if [[ "$VUHO_BUNDLE_MODEL" == "1" ]]; then
    info "Building shape: model-embedded bundle (VUHO_BUNDLE_MODEL=1) — ~500 MB, offline from first launch."

    if [[ ! -d "$VUHO_MODEL_DIR" ]]; then
        die "Model directory not found at $VUHO_MODEL_DIR (set VUHO_MODEL_DIR)"
    fi

    for comp in "${REQUIRED_COMPONENTS[@]:-}"; do
        if [[ ! -e "$VUHO_MODEL_DIR/$comp" ]]; then
            die "Missing model component: $comp (expected in $VUHO_MODEL_DIR)"
        fi
    done
else
    info "Building shape: model-less bundle (VUHO_BUNDLE_MODEL=0) — ≈40 MB on disk / ≈15 MB gzipped release tarball, model fetched on first run."
fi

# ── Build release artifacts ────────────────────────────────────────────

# Always invoke cargo, rather than skipping the build whenever $SRC_BIN
# merely *exists*. cargo's own incremental build already tracks source
# freshness correctly (mtimes/hashes) and no-ops quickly when nothing
# changed — an existence-only check here bypassed that and could bundle a
# stale target/release/vuho (built from an older checkout) stamped with the
# current commit's version, i.e. an .app whose declared version doesn't
# match the binary actually inside it.
info "Building release binary (cargo build is a fast no-op if already up to date)..."
cargo build --release -p vuho-ui --manifest-path "$WORKSPACE_ROOT/Cargo.toml"

[[ -f "$SRC_BIN" ]] || die "Release binary not found at $SRC_BIN"

# ── Assemble bundle ────────────────────────────────────────────────────

info "Cleaning stale bundle..."
rm -rf "$APP_DIR"

info "Creating bundle skeleton..."
BUNDLE_CONTENTS="$APP_DIR/Contents"
mkdir -p "$BUNDLE_CONTENTS/MacOS"
mkdir -p "$BUNDLE_CONTENTS/Resources"

info "Copying binary..."
cp "$SRC_BIN" "$BUNDLE_CONTENTS/MacOS/vuho"
chmod 755 "$BUNDLE_CONTENTS/MacOS/vuho"

info "Resolving app version from cargo metadata..."
# The single source of truth for the app version is [workspace.package]
# version in Cargo.toml; every crate (including vuho-ui, which produces the
# binary this bundle wraps) inherits it via `version.workspace = true`. Read
# it through `cargo metadata` rather than hand-parsing Cargo.toml.
APP_VERSION="$(cargo metadata --no-deps --format-version 1 --manifest-path "$WORKSPACE_ROOT/Cargo.toml" \
    | python3 -c "import json, sys; d = json.load(sys.stdin); print(next(p['version'] for p in d['packages'] if p['name'] == 'vuho-ui'))")"
[[ -n "$APP_VERSION" ]] || die "Failed to resolve app version from cargo metadata"

# Monotonic build number: see the CFBundleVersion comment in Info.plist.
#
# Commit count, not a timestamp: monotonic along main and a property of the
# commit, so it needs no separate counter file. Falls back to 0 outside a
# git checkout (e.g. a source tarball), which is still a valid
# CFBundleVersion.
#
# `git -C "$WORKSPACE_ROOT" rev-list` walks up into an *enclosing* repo if
# $WORKSPACE_ROOT isn't itself a git worktree root (e.g. a source tree
# extracted inside an unrelated checkout) — that would silently return the
# outer repo's commit count with exit 0, bypassing the `|| echo 0`
# fallback and producing a plausible-looking but non-monotonic build
# number. Guard it: only trust the count when $WORKSPACE_ROOT actually IS
# the toplevel of whatever git repo `-C` finds.
APP_BUILD=0
if git -C "$WORKSPACE_ROOT" rev-parse --is-inside-work-tree &>/dev/null; then
    GIT_TOPLEVEL="$(git -C "$WORKSPACE_ROOT" rev-parse --show-toplevel)"
    if [[ "$GIT_TOPLEVEL" == "$WORKSPACE_ROOT" ]]; then
        APP_BUILD="$(git -C "$WORKSPACE_ROOT" rev-list --count HEAD)"
    else
        info "WARNING: $WORKSPACE_ROOT is nested inside git repo $GIT_TOPLEVEL — not trusting its commit count; CFBundleVersion=0"
    fi
fi

info "Installing Info.plist (version=$APP_VERSION, build=$APP_BUILD)..."
# plutil -replace on a copy of the template, not sed: sed's replacement
# text is interpreted (an unescaped `&` in $APP_VERSION or $APP_BUILD would
# expand to the whole matched pattern, e.g. APP_BUILD='a&b' silently
# produces "a__BUILD__b"), and plutil operates on the parsed plist value
# instead of matching template text, so that whole class of bug can't
# recur.
cp "$SRC_PLIST" "$BUNDLE_CONTENTS/Info.plist"
plutil -replace CFBundleShortVersionString -string "$APP_VERSION" "$BUNDLE_CONTENTS/Info.plist"
plutil -replace CFBundleVersion -string "$APP_BUILD" "$BUNDLE_CONTENTS/Info.plist"

info "Verifying installed Info.plist's CFBundleShortVersionString and CFBundleVersion..."
INSTALLED_VERSION="$(plutil -extract CFBundleShortVersionString raw "$BUNDLE_CONTENTS/Info.plist")"
if [[ "$INSTALLED_VERSION" != "$APP_VERSION" ]]; then
    die "Installed Info.plist CFBundleShortVersionString ($INSTALLED_VERSION) != cargo metadata version ($APP_VERSION) — substitution failed"
fi
INSTALLED_BUILD="$(plutil -extract CFBundleVersion raw "$BUNDLE_CONTENTS/Info.plist")"
if [[ "$INSTALLED_BUILD" != "$APP_BUILD" ]]; then
    die "Installed Info.plist CFBundleVersion ($INSTALLED_BUILD) != resolved build number ($APP_BUILD) — substitution failed"
fi

info "Installing app icon..."
cp "$SRC_ICON" "$BUNDLE_CONTENTS/Resources/Vuho.icns"

if [[ "$VUHO_BUNDLE_MODEL" == "1" ]]; then
    info "Copying model ($MODEL_NAME)..."
    cp -R "$VUHO_MODEL_DIR" "$BUNDLE_CONTENTS/Resources/$MODEL_NAME"
else
    info "Skipping model copy (VUHO_BUNDLE_MODEL=0) — app will fetch it on first run."
fi

info "Installing ATTRIBUTION.txt..."
cp "$SRC_ATTRIBUTION" "$BUNDLE_CONTENTS/Resources/ATTRIBUTION.txt"

# ── Code-sign ──────────────────────────────────────────────────────────

# Use SIGN_ID if set (stable signature for TCC), otherwise ad-hoc (-).
if [[ -n "$SIGN_ID" ]]; then
    SIGN_ARG="$SIGN_ID"
else
    SIGN_ARG="-"
fi

info "Code-signing app bundle (sign=$SIGN_ARG)..."
codesign --force --deep --sign "$SIGN_ARG" --options runtime \
    --entitlements "$SRC_ENTITLEMENTS" \
    --timestamp=none \
    "$APP_DIR"

# ── Verification ───────────────────────────────────────────────────────

info "Verifying signature..."
codesign --verify --deep --strict --verbose=2 "$APP_DIR" || die "Code-sign verification failed"

info "Verifying plist..."
plutil -lint "$BUNDLE_CONTENTS/Info.plist" || die "Plist validation failed"

info "Verifying ort is statically linked (no bundled onnxruntime dylib)..."
if otool -L "$BUNDLE_CONTENTS/MacOS/vuho" | grep -viE '/usr/lib|/System' | grep -q dylib; then
    otool -L "$BUNDLE_CONTENTS/MacOS/vuho" >&2
    die "binary links a non-system dylib — ort was expected to be statically linked"
fi

info "Bundle size:"
du -sh "$APP_DIR"

echo ""
info "Done. Run with: open $APP_DIR"
echo "   Or test the binary directly: $BUNDLE_CONTENTS/MacOS/vuho --help"
echo ""
info "TCC permissions: on first launch, macOS will prompt for"
info "  Microphone, Accessibility, and Input Monitoring access."
if [[ -z "$SIGN_ID" ]]; then
    info "  Ad-hoc signing (--sign -) resets TCC grants on every rebuild."
    info "  Mitigation: SIGN_ID=\"Vuho Dev\" ./scripts/bundle-macos.sh"
fi
