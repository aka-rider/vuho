#!/usr/bin/env bash
# verify-app.sh — Verify Vuho.app bundle and provisioned model.
#
# Checks:
#   1. Model layout: every VUHO_BUNDLE_MODELS model's components in models/ (skipped
#      when VUHO_BUNDLE_MODEL=0 — a model-less bundle has no workspace model
#      dependency)
#   2. Bundle exists: Vuho.app binary is executable
#   3. Code signature: codesign --verify --deep --strict passes
#   4. No third-party dylibs: all dependencies are system libraries
#   5. Bundled model: every VUHO_BUNDLE_MODELS model is present in app
#      Contents/Resources/ (skipped
#      when VUHO_BUNDLE_MODEL=0 — the app fetches it on first run instead)
#   6. Attribution: ATTRIBUTION.txt file is present (warn-only)
#   7. Info.plist: plutil -lint passes, NSMicrophoneUsageDescription present,
#      CFBundleShortVersionString and CFBundleVersion are present and were
#      actually substituted (not left as the __VERSION__/__BUILD__
#      template placeholders), and Contents/Resources/Vuho.icns (the file
#      CFBundleIconFile names) exists
#
# Usage:
#   ./scripts/verify-app.sh                 # checks $ROOT/Vuho.app
#   ./scripts/verify-app.sh /path/to/app    # checks custom app path
#   VUHO_BUNDLE_MODEL=0 ./scripts/verify-app.sh  # verify a model-less bundle
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more checks failed

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────
#
# The model directory names and required-component lists come from
# models.manifest.json (repo root) — the single source of truth shared with
# the Rust build (vuho-model-paths) and the other 3 scripts.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP="${1:-$ROOT_DIR/Vuho.app}"
MANIFEST="$ROOT_DIR/models.manifest.json"
VUHO_BUNDLE_MODEL="${VUHO_BUNDLE_MODEL:-1}"

. "$SCRIPT_DIR/manifest-lib.sh"

die() { echo "ERROR: $*" >&2; exit 1; }

manifest_out=$(manifest_vars "$MANIFEST" '
emit("DEFAULT_MODEL", manifest["stt"]["default_model"])
') || die "failed to read $MANIFEST (see traceback above)"
eval "$manifest_out"

# Which models the bundle is expected to embed — the same list, with the
# same default, bundle-macos.sh copied in.
VUHO_BUNDLE_MODELS="${VUHO_BUNDLE_MODELS:-$DEFAULT_MODEL}"

# model_vars <model-id> — emits MODEL_NAME + REQUIRED_COMPONENTS for one model.
model_vars() {
    manifest_vars "$MANIFEST" "
model = manifest['stt']['models'].get('$1')
if model is None:
    raise SystemExit('unknown model id: $1')
emit('MODEL_NAME', model['dir_name'])
emit_array('REQUIRED_COMPONENTS', sorted(model['assets'].values()))
"
}

fail_count=0

# ── Helpers ─────────────────────────────────────────────────────────────

pass() { echo "[ok] $*"; }
fail() { echo "[FAIL] $*"; fail_count=$((fail_count + 1)); }
warn() { echo "[warn] $*"; }

# ── Check 1: Model layout ────────────────────────────────────────────

if [[ "$VUHO_BUNDLE_MODEL" == "1" ]]; then
    for model_id in $VUHO_BUNDLE_MODELS; do
        vars=$(model_vars "$model_id") || die "failed to read model $model_id from $MANIFEST"
        eval "$vars"
        for comp in "${REQUIRED_COMPONENTS[@]:-}"; do
            if [[ -e "$ROOT_DIR/models/$MODEL_NAME/$comp" ]]; then
                pass "Model component: $MODEL_NAME/$comp"
            else
                fail "Model component missing: $MODEL_NAME/$comp"
            fi
        done
    done
else
    warn "Skipping workspace model-layout check (VUHO_BUNDLE_MODEL=0)"
fi

# ── Check 2: Bundle exists ───────────────────────────────────────────

if ! { [[ -f "$APP/Contents/MacOS/vuho" ]] && [[ -x "$APP/Contents/MacOS/vuho" ]]; }; then
    fail "Bundle executable not found"
    echo "Hint: Run ./scripts/package.sh to build the app bundle"

    # Print bundle size if partial bundle exists
    if [[ -d "$APP" ]]; then
        echo ""
        du -sh "$APP"
    fi

    # Exit early (skip checks 3-7)
    echo ""
    if [[ $fail_count -eq 1 ]]; then
        echo "FAIL ($fail_count check failed)"
    else
        echo "FAIL ($fail_count checks failed)"
    fi
    exit 1
fi

pass "Bundle executable present"

# ── Check 3: Code signature ──────────────────────────────────────────

if codesign --verify --deep --strict "$APP" >/dev/null 2>&1; then
    pass "Code signature valid"
else
    fail "Code signature invalid"
fi

# ── Check 4: No third-party dylibs ───────────────────────────────────

bad_dylibs=$(otool -L "$APP/Contents/MacOS/vuho" | grep -viE '/usr/lib|/System' | grep dylib || true)
if [[ -z "$bad_dylibs" ]]; then
    pass "No third-party dylibs"
else
    fail "Third-party dylibs found"
fi

# ── Check 5: Bundled model ───────────────────────────────────────────

for model_id in $VUHO_BUNDLE_MODELS; do
    vars=$(model_vars "$model_id") || die "failed to read model $model_id from $MANIFEST"
    eval "$vars"
    bundled_model="$APP/Contents/Resources/$MODEL_NAME"
    if [[ "$VUHO_BUNDLE_MODEL" == "1" ]]; then
        for comp in "${REQUIRED_COMPONENTS[@]:-}"; do
            if [[ -e "$bundled_model/$comp" ]]; then
                pass "Bundled model component: $MODEL_NAME/$comp"
            else
                fail "Bundled model component missing: $MODEL_NAME/$comp"
            fi
        done
    elif [[ -e "$bundled_model" ]]; then
        fail "Model-less bundle (VUHO_BUNDLE_MODEL=0) unexpectedly contains $bundled_model"
    else
        pass "Model-less bundle correctly omits $MODEL_NAME"
    fi
done

# ── Check 6: Attribution ─────────────────────────────────────────────

if [[ -f "$APP/Contents/Resources/ATTRIBUTION.txt" ]]; then
    pass "ATTRIBUTION.txt present"
else
    warn "ATTRIBUTION.txt missing"
fi

# ── Check 7: Info.plist sanity ───────────────────────────────────────

if plutil -lint "$APP/Contents/Info.plist" >/dev/null 2>&1; then
    pass "Info.plist valid"
else
    fail "Info.plist invalid"
fi

if grep -q "NSMicrophoneUsageDescription" "$APP/Contents/Info.plist"; then
    pass "NSMicrophoneUsageDescription present"
else
    fail "NSMicrophoneUsageDescription missing"
fi

plist_short_version="$(plutil -extract CFBundleShortVersionString raw "$APP/Contents/Info.plist" 2>/dev/null || true)"
if [[ -n "$plist_short_version" && "$plist_short_version" != "__VERSION__" ]]; then
    pass "CFBundleShortVersionString present ($plist_short_version)"
else
    fail "CFBundleShortVersionString missing or unsubstituted (__VERSION__ placeholder left in place)"
fi

plist_build="$(plutil -extract CFBundleVersion raw "$APP/Contents/Info.plist" 2>/dev/null || true)"
if [[ -n "$plist_build" && "$plist_build" != "__BUILD__" ]]; then
    pass "CFBundleVersion present ($plist_build)"
else
    fail "CFBundleVersion missing or unsubstituted (__BUILD__ placeholder left in place)"
fi

# CFBundleIconFile names "Vuho" (no extension) per Apple convention — the
# actual file on disk is Vuho.icns.
if [[ -f "$APP/Contents/Resources/Vuho.icns" ]]; then
    pass "Vuho.icns present (CFBundleIconFile target)"
else
    fail "Vuho.icns missing — CFBundleIconFile names it but Contents/Resources/Vuho.icns is absent"
fi

# ── Bundle size ──────────────────────────────────────────────────────

echo ""
du -sh "$APP"

# ── Summary ──────────────────────────────────────────────────────────

echo ""
if [[ $fail_count -eq 0 ]]; then
    echo "PASS"
    exit 0
else
    if [[ $fail_count -eq 1 ]]; then
        echo "FAIL ($fail_count check failed)"
    else
        echo "FAIL ($fail_count checks failed)"
    fi
    exit 1
fi
