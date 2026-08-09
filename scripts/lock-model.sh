#!/usr/bin/env bash
# lock-model.sh — Record one STT model's per-file SHA-256 + size into
# models.lock.json, cross-checked against the pinned Hugging Face revision
# before anything is written.
#
# The write is a read-modify-write keyed by model id: every other model's
# entry in the lock is carried over untouched, so locking one model never
# invalidates another's already-verified hashes.
#
# Usage:
#   ./scripts/lock-model.sh <model-id>
#
# Prerequisites:
#   - The model must already be provisioned locally (./scripts/fetch-model.sh)
#   - python3 (for JSON emission, hashing, and the Hugging Face API cross-check)
#
# The upstream repo, pinned revision, model dir name, and asset list all
# come from models.manifest.json (repo root) via scripts/manifest-lib.sh —
# the single source of truth shared with the Rust build (vuho-model-paths)
# and the other provisioning scripts. This script walks ONLY the model's
# manifested `assets`, so any unmanifested directory
# sitting in the local checkout (e.g. Melspectrogram_15s.mlmodelc) is
# skipped.
#
# It hashes local file content — deliberately NOT Hugging Face's `lfs.oid`:
# only a subset of the repo's files are LFS/Xet-backed and expose a sha256;
# the rest (every model.mil, every metadata.json) carry only a git SHA-1
# blob OID. Hashing local content keeps verification uniform. But hashing
# one developer's disk only pins whatever happens to be on it, so before
# writing the lock this script fetches the pinned revision's file tree from
# the Hugging Face API and asserts: the local component file set is exactly
# the remote component file set (nothing missing, nothing extra), every
# file's size matches, and — for the files that expose one — the remote
# `lfs.oid` equals the locally computed SHA-256. Any mismatch is a hard
# failure naming the offending file and both values; no lock file is
# written.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
MANIFEST="$ROOT_DIR/models.manifest.json"
LOCK_FILE="$ROOT_DIR/models.lock.json"

# Timeout (seconds) for each Hugging Face tree-API request.
readonly HF_API_TIMEOUT_SECS=30

# Lock schema version this script emits — must match `vuho_model_paths::Lock`'s
# `schema_version` assertion.
readonly LOCK_SCHEMA_VERSION=2

log() { printf "[lock-model] %s\n" "$*"; }
die() { printf "[lock-model] ERROR: %s\n" "$*" >&2; exit 1; }

# ── Read the manifest ───────────────────────────────────────────────────

. "$SCRIPT_DIR/manifest-lib.sh"

MODEL_ID="${1:-}"
[[ -n "$MODEL_ID" ]] || die "usage: $0 <model-id>"

manifest_out=$(manifest_vars "$MANIFEST" "
model = manifest['stt']['models'].get('$MODEL_ID')
if model is None:
    raise SystemExit('unknown model id: $MODEL_ID (known: ' + ', '.join(sorted(manifest['stt']['models'])) + ')')
emit('STT_REPO', model['repo'])
emit('STT_REV', model['revision'])
emit('STT_DIR_NAME', model['dir_name'])
emit_array('STT_COMPONENTS', sorted(model['assets'].values()))
") || die "failed to read $MANIFEST (see traceback above)"
eval "$manifest_out"

MODEL_DIR="$MODELS_DIR/$STT_DIR_NAME"

if [[ ! -d "$MODEL_DIR" ]]; then
    log "ERROR: model dir not found: $MODEL_DIR (run ./scripts/fetch-model.sh first)"
    exit 1
fi

log "Locking $MODEL_ID ($STT_REPO@$STT_REV) from $MODEL_DIR"

# ── Walk local components, hash, cross-check against HF, emit the lock ──

python3 - "$MODEL_ID" "$STT_REPO" "$STT_REV" "$STT_DIR_NAME" "$MODEL_DIR" "$LOCK_FILE" "$HF_API_TIMEOUT_SECS" "$LOCK_SCHEMA_VERSION" "${STT_COMPONENTS[@]}" <<'PY'
import hashlib
import json
import sys
import urllib.error
import urllib.request

model_id, repo, rev, dir_name, model_dir, lock_file, timeout_str, schema_version_str = sys.argv[1:9]
components = sys.argv[9:]
timeout = float(timeout_str)
schema_version = int(schema_version_str)


def fail(message: str) -> None:
    print(f"[lock-model] ERROR: {message}", file=sys.stderr)
    sys.exit(1)


# ── 1. Walk the local tree, scoped to the manifest's components ─────────

import os

local_files = {}  # rel_path -> (size, sha256)

for comp in components:
    comp_path = os.path.join(model_dir, comp)
    if os.path.isdir(comp_path):
        for root, _dirs, files in os.walk(comp_path):
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, model_dir).replace(os.sep, "/")
                local_files[rel] = full
    elif os.path.isfile(comp_path):
        local_files[comp] = comp_path
    else:
        fail(f"manifest component missing locally: {comp} (expected at {comp_path})")

hashes = {}
sizes = {}
for rel, full in local_files.items():
    h = hashlib.sha256()
    size = 0
    with open(full, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    hashes[rel] = h.hexdigest()
    sizes[rel] = size

local_paths = set(local_files.keys())

# ── 2. Fetch the pinned revision's file tree from the HF API ────────────

def fetch_tree(repo: str, rev: str) -> list:
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
    entries = []
    while url:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                page = json.loads(resp.read().decode("utf-8"))
                link_header = resp.headers.get("Link")
        except urllib.error.URLError as e:
            fail(f"failed to fetch HF tree API ({url}): {e}")
        entries.extend(page)
        # Follow RFC 5988 pagination if the API paginates this repo's tree.
        url = None
        if link_header:
            for part in link_header.split(","):
                segs = part.split(";")
                if len(segs) >= 2 and 'rel="next"' in segs[1]:
                    url = segs[0].strip().lstrip("<").rstrip(">")
                    break
    return entries

tree = fetch_tree(repo, rev)

component_set = set(components)


def in_scope(path: str) -> bool:
    top = path.split("/", 1)[0]
    return top in component_set


remote_files = {
    e["path"]: e
    for e in tree
    if e.get("type") == "file" and in_scope(e["path"])
}
remote_paths = set(remote_files.keys())

# ── 3. Cross-check: file set, then size, then lfs.oid where present ─────

missing_locally = remote_paths - local_paths
extra_locally = local_paths - remote_paths
if missing_locally or extra_locally:
    details = []
    if missing_locally:
        details.append(
            "present at HF revision but missing locally: "
            + ", ".join(sorted(missing_locally))
        )
    if extra_locally:
        details.append(
            "present locally but not at HF revision (or out of component scope): "
            + ", ".join(sorted(extra_locally))
        )
    fail("local component file set does not match the pinned HF revision — " + "; ".join(details))

lfs_checked = 0
size_only = 0

for path in sorted(remote_paths):
    remote_entry = remote_files[path]
    remote_size = remote_entry.get("size")
    local_size = sizes[path]
    if remote_size != local_size:
        fail(
            f"size mismatch for {path}: local={local_size} bytes, "
            f"HF revision {rev}={remote_size} bytes"
        )

    lfs = remote_entry.get("lfs")
    if lfs and lfs.get("oid"):
        remote_oid = lfs["oid"]
        local_sha256 = hashes[path]
        if remote_oid != local_sha256:
            fail(
                f"sha256 mismatch for {path}: local={local_sha256}, "
                f"HF lfs.oid at revision {rev}={remote_oid}"
            )
        lfs_checked += 1
    else:
        size_only += 1

print(
    f"[lock-model] cross-checked {len(remote_paths)} files against HF revision {rev}: "
    f"{lfs_checked} via lfs.oid, {size_only} via size-only "
    "(no LFS/Xet sha256 exposed for these — every file's local content was still hashed "
    "into the lock)."
)

# ── 4. Merge this model into the lock, sorted by path for a stable diff ──

files = [
    {"path": path, "size": sizes[path], "sha256": hashes[path]}
    for path in sorted(local_paths)
]
total_bytes = sum(f["size"] for f in files)

try:
    with open(lock_file, encoding="utf-8") as f:
        lock = json.load(f)
except FileNotFoundError:
    lock = {"schema_version": schema_version, "models": {}}

if lock.get("schema_version") != schema_version:
    fail(
        f"{lock_file} has schema_version {lock.get('schema_version')}, "
        f"this script writes {schema_version} — migrate it before re-locking, "
        "rather than silently mixing two shapes in one file"
    )

lock["models"][model_id] = {
    "dir_name": dir_name,
    "revision": rev,
    "total_bytes": total_bytes,
    "files": files,
}
lock["models"] = dict(sorted(lock["models"].items()))

with open(lock_file, "w", encoding="utf-8") as f:
    json.dump(lock, f, indent=2, sort_keys=False)
    f.write("\n")

print(
    f"[lock-model] wrote {lock_file}: {model_id} = {len(files)} files, {total_bytes} bytes "
    f"({len(lock['models'])} models locked)"
)
PY

log "Done."
