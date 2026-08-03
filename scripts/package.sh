#!/usr/bin/env bash
# package.sh — Build and package Vuho as a signed .app.
#
# This is the entry-point script for the packaging pipeline.
# It delegates to bundle-macos.sh after building release artifacts.
#
# Usage:
#   ./scripts/package.sh
#
# TCC developer workflow (ad-hoc signing):
#   Every ad-hoc rebuild produces a new signature → TCC re-grant required.
#   For iterative development, create a self-signed cert:
#     1. Keychain Access → Certificate Assistant → Create a Certificate...
#        Name: "Vuho Dev", Identity Type: Self-Signed Root, Certificate Type: Code Signing
#     2. Re-run with: SIGN_ID="<cert-common-name>" ./scripts/package.sh
#        (set SIGN_ID env var to avoid TCC re-grants across rebuilds)
#   Or reset TCC grants in one command (macOS Ventura+), using the bundle ID
#   from models.manifest.json's "bundle_id" field (<bundle-id> below):
#     tccutil reset Microphone <bundle-id>
#     tccutil reset Accessibility <bundle-id>
#     tccutil reset InputMonitoring <bundle-id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export SIGN_ID="${SIGN_ID:-}"

if [[ -n "$SIGN_ID" ]]; then
    echo "==> Using signing identity: $SIGN_ID"
    export SIGN_ID
else
    echo "==> Using ad-hoc signing (--sign -)"
fi

exec bash "$SCRIPT_DIR/bundle-macos.sh"
