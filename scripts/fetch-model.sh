#!/usr/bin/env bash
# fetch-model.sh — Download the parakeet TDT + Silero VAD models.
#
# Idempotent: safe to re-run. A file that is already present *and verified
# complete* is skipped; a missing, truncated, or corrupt file is (re)fetched
# — see "Resumability" below for how that's made true rather than aspirational.
#
# Usage:
#   ./scripts/fetch-model.sh          # download everything
#   ./scripts/fetch-model.sh parakeet # download only the parakeet model
#   ./scripts/fetch-model.sh silero   # download only Silero VAD
#
# Prerequisites:
#   - `huggingface-cli` (from `pip install huggingface_hub`) — preferred
#   - Falls back to `curl` when the CLI is unavailable, OR when it is
#     present but fails (offline, HF 401/429, disk full, etc.) — see
#     download_file() below.
#
# Model names, upstream repos, pinned revisions, and component lists all
# come from `models.manifest.json` (repo root) — the single source of truth
# shared with the Rust build (`vuho-model-paths`) and the other 3 scripts.
# `jfk.wav` (the `test-stt-ffi` regression fixture) is tracked directly in
# git, not fetched here — it has no canonical upstream source to pin.
#
# After fetching the parakeet model, the resulting tree is cross-checked
# against `models.lock.json` (repo root, produced by `scripts/lock-model.sh`)
# — every locked file must be present at its locked size *and* sha256, or
# the script fails loudly naming exactly what's missing/mismatched. This
# exists because the Hugging Face tree API used by the curl fallback below
# is non-recursive by default: a `.mlmodelc` bundle nests real content one
# level deeper (e.g. `weights/weight.bin`, `analytics/coremldata.bin`), and
# without `?recursive=true` those files are silently never listed, so a
# partial tree was fetched with a zero exit code and nothing noticed.
# models.lock.json ships in the repo (it is not optional/best-effort), so
# its absence is itself a hard failure, not a skipped check.
#
# Resumability / corruption safety: every download (huggingface-cli AND the
# curl fallback, both the single-file and recursive-directory paths) writes
# to a scratch path first and only `mv`s it into place after it completes
# successfully. An interrupted transfer therefore never leaves a truncated
# file sitting at the real destination for a later run to mistake for
# "already downloaded" — the on-disk state is always either "absent" or
# "complete," never "partial-but-looks-done." Scratch paths live under one
# `mktemp -d` root cleaned up by an EXIT trap, so a killed run doesn't leak
# `.tmp-*` directories under models/ either.
#
# Hashing cost: the completeness check reads and SHA-256-hashes the full
# ~474 MB parakeet tree every run (~1-2s on this machine) — deliberate, not
# an oversight. A size-only check passes same-length corruption; this
# script exists specifically to catch corruption the naive check missed.
# Set VUHO_SKIP_HASH_VERIFY=1 to fall back to the cheaper size-only check
# for rapid local iteration when you trust the tree wasn't corrupted since
# the last full verify.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
MANIFEST="$ROOT_DIR/models.manifest.json"
LOCK_FILE="$ROOT_DIR/models.lock.json"
VUHO_SKIP_HASH_VERIFY="${VUHO_SKIP_HASH_VERIFY:-0}"

# curl retry policy for both the plain-file and recursive-listing fetch paths.
readonly CURL_RETRY_COUNT=3
readonly CURL_RETRY_DELAY_SECS=5

# ── Helpers ───────────────────────────────────────────────────────────

log() { printf "[fetch-model] %s\n" "$*"; }
die() { printf "[fetch-model] ERROR: %s\n" "$*" >&2; exit 1; }

# One scratch root for every in-flight download this run; removed on exit
# (success, failure, or signal) so nothing leaks under models/.
TMP_ROOT="$(mktemp -d "$MODELS_DIR/.fetch-model.XXXXXX" 2>/dev/null || { mkdir -p "$MODELS_DIR"; mktemp -d "$MODELS_DIR/.fetch-model.XXXXXX"; })"
cleanup() { rm -rf "$TMP_ROOT"; }
trap cleanup EXIT

# ── Read the manifest ───────────────────────────────────────────────────

. "$SCRIPT_DIR/manifest-lib.sh"

manifest_out=$(manifest_vars "$MANIFEST" '
stt = manifest["stt"]
silero = manifest["silero"]

emit("PARAKEET_REPO", stt["repo"])
emit("PARAKEET_REV", stt["revision"])
emit("PARAKEET_DIR_NAME", stt["dir_name"])
emit_array("PARAKEET_COMPONENTS", stt["components"])

emit("SILERO_REPO", silero["repo"])
emit("SILERO_REV", silero["revision"])
emit("SILERO_DIR_NAME", silero["dir_name"])
emit_array("SILERO_COMPONENTS", silero["components"])
') || die "failed to read $MANIFEST (see traceback above)"
eval "$manifest_out"

PARAKEET_DIR="$MODELS_DIR/$PARAKEET_DIR_NAME"
SILERO_DIR="$MODELS_DIR/$SILERO_DIR_NAME"

# Try huggingface-cli first, fall back to curl — including when the CLI is
# *present but fails* (offline, HF 401/429, disk full): both stdout and
# stderr are captured so a CLI failure prints a real diagnostic instead of
# being silently discarded, and the fallback only skips a destination once
# it exists AND is non-empty (an interrupted `mv` never leaves a 0-byte file
# behind, but be defensive about anything else that might).
# download_file <repo> <rev> <path_in_repo> <dest_path>
download_file() {
    local repo="$1" rev="$2" path="$3" dest="$4"

    if [[ -s "$dest" ]]; then
        log "  skip  $dest (exists)"
        return 0
    fi

    if command -v huggingface-cli &>/dev/null; then
        local cli_scratch cli_log
        cli_scratch="$TMP_ROOT/cli-$(basename "$dest")"
        cli_log="$TMP_ROOT/cli-$(basename "$dest").log"
        if huggingface-cli download "$repo" "$path" --revision "$rev" --local-dir "$cli_scratch" >"$cli_log" 2>&1; then
            if [[ -s "$cli_scratch/$path" ]]; then
                mkdir -p "$(dirname "$dest")"
                mv "$cli_scratch/$path" "$dest"
                return 0
            fi
            log "  WARNING: huggingface-cli reported success but $path is missing/empty in its output — falling back to curl"
        else
            log "  WARNING: huggingface-cli failed for $path (falling back to curl); output:"
            sed 's/^/  |  /' "$cli_log" >&2
        fi
    fi

    # Curl fallback: resolve the download URL from the commit-pinned raw URL.
    # Download to a scratch file and rename into place, so an interrupted
    # transfer never leaves a truncated file at $dest for a later run to
    # mistake for "already downloaded."
    local url="https://huggingface.co/${repo}/resolve/${rev}/${path}"
    local scratch
    scratch="$TMP_ROOT/curl-$(basename "$dest")"
    log "  curl  $url → $dest"
    mkdir -p "$(dirname "$dest")"
    if ! curl -fSL --retry "$CURL_RETRY_COUNT" --retry-delay "$CURL_RETRY_DELAY_SECS" -o "$scratch" "$url"; then
        log "  ERROR: failed to download $path from $repo"
        return 1
    fi
    mv "$scratch" "$dest"
}

# List every *file* (not directory) nested anywhere under a repo path, via
# the Hugging Face tree API's `?recursive=true` — required because a
# `.mlmodelc` bundle nests content (weights/, analytics/) one level below
# its top-level files, and the non-recursive listing only reports those
# subdirectories' names, never their contents.
# NOTE: unlike scripts/lock-model.sh's equivalent Python fetch_tree(), this
# does NOT follow RFC 5988 `Link: rel="next"` pagination — a single-page
# request only. Accepted, not fixed: each of the manifest's components
# (Preprocessor/Encoder/Decoder/RNNTJoint/.mlmodelc dirs) lists on the order
# of 10-20 files, far under whatever page size would trigger pagination.
# Revisit if a future component ever approaches that (the two scripts would
# then silently disagree: lock-model.sh would see every file, this would
# see only the first page).
# hf_list_files_recursive <repo> <rev> <path_in_repo>
hf_list_files_recursive() {
    local repo="$1" rev="$2" path="$3"
    local listing
    listing=$(curl -fsSL "https://huggingface.co/api/models/${repo}/tree/${rev}/${path}?recursive=true" 2>/dev/null || true)
    if [[ -z "$listing" || "$listing" == "null" ]]; then
        return 1
    fi
    echo "$listing" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data:
    if f.get('type') == 'file':
        print(f['path'])
"
}

# Download every file listed under a repo directory (recursively), skipping
# any destination that already exists and is non-empty. Each file is
# written to a scratch path and renamed into place, same rationale as
# download_file() above.
# fetch_dir_recursive <repo> <rev> <path_in_repo> <local_root>
fetch_dir_recursive() {
    local repo="$1" rev="$2" path="$3" local_root="$4"
    local files
    if ! files=$(hf_list_files_recursive "$repo" "$rev" "$path"); then
        log "  ERROR: failed to list $path via the HF tree API"
        return 1
    fi
    if [[ -z "$files" ]]; then
        log "  ERROR: HF tree API returned no files for $path"
        return 1
    fi

    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        local dest="$local_root/$f"
        if [[ -s "$dest" ]]; then
            log "  skip  $dest (exists)"
            continue
        fi
        mkdir -p "$(dirname "$dest")"
        local url="https://huggingface.co/${repo}/resolve/${rev}/${f}"
        local scratch
        scratch="$TMP_ROOT/dir-$(echo "$f" | tr '/' '_')"
        log "  curl  $f → $dest"
        curl -fSL --retry "$CURL_RETRY_COUNT" --retry-delay "$CURL_RETRY_DELAY_SECS" -o "$scratch" "$url" || {
            log "  ERROR: failed to download $f"
            return 1
        }
        mv "$scratch" "$dest"
    done <<< "$files"
}

# ── Parakeet model ────────────────────────────────────────────────────

fetch_parakeet() {
    log "Downloading parakeet TDT v3 model from $PARAKEET_REPO@$PARAKEET_REV"
    mkdir -p "$PARAKEET_DIR"

    for comp in "${PARAKEET_COMPONENTS[@]:-}"; do
        if [[ "$comp" == *.mlmodelc ]]; then
            # .mlmodelc is a directory, recursively populated (weights/,
            # analytics/) — walk and fetch it via the HF tree API.
            fetch_dir_recursive "$PARAKEET_REPO" "$PARAKEET_REV" "$comp" "$PARAKEET_DIR"
        else
            # Regular file (vocab.json)
            download_file "$PARAKEET_REPO" "$PARAKEET_REV" "$comp" "$PARAKEET_DIR/$comp"
        fi
    done

    log "Parakeet model ready: $PARAKEET_DIR"
    assert_parakeet_complete
}

# Cross-check the fetched parakeet tree against models.lock.json: every
# locked file must be present locally at its locked size and (unless
# VUHO_SKIP_HASH_VERIFY=1) sha256. This is the safety net for the bug
# documented at the top of this file — a script bug (or a future upstream
# repo change) that silently drops or truncates files now fails loudly
# instead of shipping a model that loads but produces wrong or crashing
# inference. models.lock.json ships in the repo, so its absence is a hard
# failure — never a silently-skipped warning.
assert_parakeet_complete() {
    if [[ ! -f "$LOCK_FILE" ]]; then
        die "$LOCK_FILE not found — it ships in the repo; a missing lock means this checkout is broken, not that verification is optional. Restore it (git checkout $LOCK_FILE) or regenerate it (./scripts/lock-model.sh) before trusting the fetched model."
    fi

    local lock_out
    lock_out=$(manifest_vars "$LOCK_FILE" '
stt = manifest["stt"]
emit_array("LOCK_PATHS", [f["path"] for f in stt["files"]])
emit_array("LOCK_SIZES", [str(f["size"]) for f in stt["files"]])
emit_array("LOCK_SHA256S", [f["sha256"] for f in stt["files"]])
') || die "failed to read $LOCK_FILE (see traceback above)"
    eval "$lock_out"

    local missing=() mismatched=()
    local i path expected_size actual_size expected_sha actual_sha dest
    for ((i = 0; i < ${#LOCK_PATHS[@]}; i++)); do
        path="${LOCK_PATHS[$i]}"
        expected_size="${LOCK_SIZES[$i]}"
        dest="$PARAKEET_DIR/$path"
        if [[ ! -f "$dest" ]]; then
            missing+=("$path")
            continue
        fi
        actual_size=$(wc -c < "$dest" | tr -d ' ')
        if [[ "$actual_size" != "$expected_size" ]]; then
            mismatched+=("$path (expected ${expected_size}B, got ${actual_size}B)")
            continue
        fi
        if [[ "$VUHO_SKIP_HASH_VERIFY" != "1" ]]; then
            expected_sha="${LOCK_SHA256S[$i]}"
            actual_sha=$(shasum -a 256 "$dest" | awk '{print $1}')
            if [[ "$actual_sha" != "$expected_sha" ]]; then
                mismatched+=("$path (sha256 expected $expected_sha, got $actual_sha)")
            fi
        fi
    done

    if (( ${#missing[@]} > 0 || ${#mismatched[@]} > 0 )); then
        log "ERROR: parakeet model tree is INCOMPLETE against $LOCK_FILE"
        for path in "${missing[@]:-}"; do
            [[ -z "$path" ]] && continue
            log "  MISSING        $path"
        done
        for entry in "${mismatched[@]:-}"; do
            [[ -z "$entry" ]] && continue
            log "  MISMATCH       $entry"
        done
        log "Delete the listed files and re-run this script to repair them (downloads skip only files that already exist)."
        return 1
    fi

    if [[ "$VUHO_SKIP_HASH_VERIFY" == "1" ]]; then
        log "Completeness check passed (size-only, VUHO_SKIP_HASH_VERIFY=1): ${#LOCK_PATHS[@]} files match $LOCK_FILE"
    else
        log "Completeness check passed (size + sha256): ${#LOCK_PATHS[@]} files match $LOCK_FILE"
    fi
}

# ── Silero VAD ────────────────────────────────────────────────────────

fetch_silero() {
    log "Downloading Silero VAD from $SILERO_REPO@$SILERO_REV"
    # Ensure the onnx/ subdirectory exists (the repo structure nests files under onnx/)
    mkdir -p "$SILERO_DIR/onnx"

    for comp in "${SILERO_COMPONENTS[@]:-}"; do
        download_file "$SILERO_REPO" "$SILERO_REV" "$comp" "$SILERO_DIR/$comp"
    done

    log "Silero VAD ready: $SILERO_DIR"
    assert_silero_complete
}

# Silero has no per-file lock (models.lock.json only covers the parakeet
# tree — a single small ONNX file doesn't warrant one), so this can only
# assert existence and non-zero size, not a hash. That is still strictly
# better than the prior no-check-at-all: download_file()'s scratch-then-
# rename means a file present at all is at least a *complete* transfer, so
# this check exists to catch a component silently missing from the
# manifest's component list rather than partial-write corruption (which the
# rename already rules out).
assert_silero_complete() {
    local comp dest missing=()
    for comp in "${SILERO_COMPONENTS[@]:-}"; do
        dest="$SILERO_DIR/$comp"
        if [[ ! -s "$dest" ]]; then
            missing+=("$comp")
        fi
    done
    if (( ${#missing[@]} > 0 )); then
        log "ERROR: Silero VAD tree is INCOMPLETE"
        for comp in "${missing[@]:-}"; do
            [[ -z "$comp" ]] && continue
            log "  MISSING/EMPTY  $comp"
        done
        return 1
    fi
    log "Completeness check passed: ${#SILERO_COMPONENTS[@]} files present and non-empty"
}

# ── Main ──────────────────────────────────────────────────────────────

main() {
    local want_parakeet=1
    local want_silero=1

    case "${1:-all}" in
        parakeet) want_parakeet=1; want_silero=0 ;;
        silero)   want_parakeet=0; want_silero=1 ;;
        all)      ;;
        *)        echo "Usage: $0 [all|parakeet|silero]"; exit 1 ;;
    esac

    if (( want_parakeet )); then fetch_parakeet; fi
    if (( want_silero ));   then fetch_silero;   fi

    log "Done."
}

main "$@"
