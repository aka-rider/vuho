# shellcheck shell=bash
# manifest-lib.sh — shared manifest-reading helper sourced (not executed)
# by the scripts that eval fields of models.manifest.json into shell
# variables. No shebang: it has no execute bit and is never run directly.
#
# SECURITY: every caller does `eval "$(manifest_vars ...)"` (or the safer
# `vars=$(manifest_vars ...) || die ...; eval "$vars"` form — see below).
# That means whatever this function prints is executed as shell code. The
# only thing standing between manifest/lock *content* (an untrusted-ish
# JSON file, however unlikely to be attacker-controlled in practice) and
# arbitrary code execution is emit()/emit_array() below routing every value
# through Python's `shlex.quote` before it reaches shell. Do not change
# emit()/emit_array() to interpolate a value into the printed line without
# quoting it, and do not add a third emit-like helper that skips it.
#
# CALLING CONVENTION: never call this as a bare `eval "$(manifest_vars ...)"`
# — `eval "$(cmd)"` with cmd failing prints nothing, `eval ""` is a no-op
# that returns 0, and the command substitution's own exit status is
# discarded. A malformed manifest or a renamed key then produces a Python
# traceback on stderr but the script *keeps running* with the variables it
# was expecting left unset, and dies several lines later on an unrelated
# `set -u` "unbound variable" that never names the manifest. Always do:
#
#   vars=$(manifest_vars "$MANIFEST" '...') || die "failed to read $MANIFEST"
#   eval "$vars"
#
# manifest_vars <manifest-file> <python-body>
#
# Runs <python-body> with the parsed manifest bound to `manifest` and the
# emit()/emit_array() helpers defined; prints NAME=value / NAME=(values)
# lines on stdout for the caller to eval.
manifest_vars() {
    python3 - "$1" "$2" <<'PY'
import json
import shlex
import sys


def emit(name: str, value: str) -> None:
    print(f"{name}={shlex.quote(value)}")


def emit_array(name: str, values: list[str]) -> None:
    quoted = " ".join(shlex.quote(v) for v in values)
    print(f"{name}=({quoted})")


with open(sys.argv[1], encoding="utf-8") as f:
    manifest = json.load(f)

exec(sys.argv[2])
PY
}
