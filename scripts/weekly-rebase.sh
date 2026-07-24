#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

usage() {
    printf 'Usage: %s [OPTIONS]\n' "${0##*/}"
    printf '\n'
    printf 'Weekly rebase wrapper for the halo branch.\n'
    printf '\n'
    printf 'Runs: scripts/rebase-halo.sh with --build --push --progress --log --state\n'
    printf '\n'
    printf 'Options:\n'
    printf '  --dry-run     Preview without executing\n'
    printf '  --skip-build  Skip ROCm build verification\n'
    printf '  --skip-push   Skip git push to origin\n'
    printf '  --skip-progress  Skip progress.md update\n'
    printf '  -h, --help    Show this help message\n'
    printf '\n'
    printf 'Example:\n'
    printf '  git checkout halo && ./scripts/weekly-rebase.sh\n'
    printf '  git checkout halo && ./scripts/weekly-rebase.sh --dry-run\n'
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    usage
    exit 0
fi

DRY_RUN=false
SKIP_BUILD=false
SKIP_PUSH=false
SKIP_PROGRESS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --skip-push) SKIP_PUSH=true; shift ;;
        --skip-progress) SKIP_PROGRESS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown option: $1 (try --help)" ;;
    esac
done

CURRENT_BRANCH=$(git symbolic-ref --quiet --short HEAD 2>/dev/null) || fail "detached HEAD is not supported"
[[ "$CURRENT_BRANCH" == "halo" ]] || fail "this script is designed to run from the halo branch (currently on $CURRENT_BRANCH)"

ARGS="--log --state"
[[ "$SKIP_BUILD" == "true" ]] || ARGS="$ARGS --build"
[[ "$SKIP_PUSH" == "true" ]] || ARGS="$ARGS --push"
[[ "$SKIP_PROGRESS" == "true" ]] || ARGS="$ARGS --progress"
[[ "$DRY_RUN" == "true" ]] && ARGS="$ARGS --dry-run"

log_file="$ROOT_DIR/.rebase-halo.log"
printf '=== weekly-rebase (%s) ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Command: scripts/rebase-halo.sh %s\n' "$ARGS"
printf 'Log file: %s\n' "$log_file"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
    scripts/rebase-halo.sh $ARGS
else
    scripts/rebase-halo.sh $ARGS
    printf '\n=== weekly-rebase complete ===\n'
fi