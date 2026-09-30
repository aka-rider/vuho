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
# manifest_host_os — prints the manifest's name for the running OS.
manifest_host_os() {
    case "$(uname -s)" in
        Darwin) echo macos ;;
        Linux) echo linux ;;
        *) echo "manifest-lib: unsupported OS $(uname -s)" >&2; return 1 ;;
    esac
}

# sha256_of <file> — prints the file's hex SHA-256 (GNU sha256sum on Linux,
# shasum on macOS).
sha256_of() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# manifest_vars <manifest-file> <python-body>
#
# Runs <python-body> with the parsed manifest bound to `manifest`, the
# running OS's manifest key (`macos` or `linux`) bound to `host_os`, and the
# emit()/emit_array() helpers defined; prints NAME=value / NAME=(values)
# lines on stdout for the caller to eval.
manifest_vars() {
    local host_os
    host_os=$(manifest_host_os) || return 1
    python3 - "$1" "$2" "$host_os" <<'PY'
import json
import shlex
import sys


def emit(name: str, value: str) -> None:
    print(f"{name}={shlex.quote(value)}")


def emit_array(name: str, values: list[str]) -> None:
    quoted = " ".join(shlex.quote(v) for v in values)
    print(f"{name}=({quoted})")


host_os = sys.argv[3]

with open(sys.argv[1], encoding="utf-8") as f:
    manifest = json.load(f)

exec(sys.argv[2])
PY
}
