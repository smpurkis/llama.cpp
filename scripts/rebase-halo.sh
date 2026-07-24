#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)
TARGET_BRANCH=${TARGET_BRANCH:-halo}
STOCK_REMOTE=${STOCK_REMOTE:-upstream}
STOCK_URL=${STOCK_URL:-https://github.com/ggml-org/llama.cpp.git}
CACHY_REMOTE=${CACHY_REMOTE:-cachyllama}
CACHY_URL=${CACHY_URL:-https://github.com/fewtarius/CachyLLama.git}
BASE_BRANCH=${BASE_BRANCH:-cachyllama-rebased}
STATE_FILE="$ROOT_DIR/.rebase-halo.state"
LOG_FILE="$ROOT_DIR/.rebase-halo.log"

cd "$ROOT_DIR"

usage() {
    printf 'Usage: %s [OPTIONS]\n' "${0##*/}"
    printf '\n'
    printf 'Layered rebase: upstream/master -> CachyLLama/master -> halo.\n'
    printf '\n'
    printf 'Options:\n'
    printf '  --build       Run ROCm build after rebase (requires ROCm toolchain)\n'
    printf '  --push        Push halo to origin after successful rebase\n'
    printf '  --progress    Update progress.md with change summary after rebase\n'
    printf '  --log         Append rebase result to rebase log file\n'
    printf '  --state       Update state file with new base commits\n'
    printf '  --dry-run     Print what would happen without running any commands\n'
    printf '  -h, --help    Show this help message\n'
    printf '\n'
    printf 'Environment variables:\n'
    printf '  TARGET_BRANCH   Target branch (default: halo)\n'
    printf '  STOCK_REMOTE    Upstream remote name (default: upstream)\n'
    printf '  STOCK_URL       Upstream remote URL (default: ggml-org/llama.cpp)\n'
    printf '  CACHY_REMOTE    CachyLLama remote name (default: cachyllama)\n'
    printf '  CACHY_URL       CachyLLama remote URL (default: fewtarius/CachyLLama)\n'
    printf '  BASE_BRANCH     Branch name for rebased CachyLLama layer (default: cachyllama-rebased)\n'
    printf '\n'
    printf 'Example (weekly rebase with build and push):\n'
    printf '  git checkout halo\n'
    printf '  scripts/rebase-halo.sh --build --push\n'
}

fail() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

ensure_remote() {
    local remote=$1
    local url=$2

    if ! git remote get-url "$remote" >/dev/null 2>&1; then
        git remote add "$remote" "$url"
    fi
}

log() {
    local msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
    printf '%s\n' "$msg"
    if [[ -n "${LOG_FILE:-}" && "${DRY_RUN:-}" != "true" ]]; then
        printf '%s\n' "$msg" >> "$LOG_FILE"
    fi
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    usage
    exit 0
fi

BUILD=false
PUSH=false
UPDATE_PROGRESS=false
UPDATE_LOG=false
UPDATE_STATE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD=true; shift ;;
        --push) PUSH=true; shift ;;
        --progress) UPDATE_PROGRESS=true; shift ;;
        --log) UPDATE_LOG=true; shift ;;
        --state) UPDATE_STATE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown option: $1 (try --help)" ;;
    esac
done

CURRENT_BRANCH=$(git symbolic-ref --quiet --short HEAD 2>/dev/null) || fail "detached HEAD is not supported"
[[ "$CURRENT_BRANCH" == "$TARGET_BRANCH" ]] || fail "check out $TARGET_BRANCH before running this script"
git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH" || fail "local branch $TARGET_BRANCH does not exist"

if [[ "$DRY_RUN" != "true" ]]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        fail "working tree has tracked changes; commit or stash them before rebasing"
    fi
fi

ensure_remote "$STOCK_REMOTE" "$STOCK_URL"
ensure_remote "$CACHY_REMOTE" "$CACHY_URL"

log "=== weekly-rebase starting for $TARGET_BRANCH ==="
log "dry-run=$DRY_RUN build=$BUILD push=$PUSH progress=$UPDATE_PROGRESS"

# Capture old SHAs before fetch so we can tell what changed
PREV_STOCK_SHA=$(git rev-parse "$STOCK_REMOTE/master" 2>/dev/null || true)
PREV_CACHY_SHA=$(git rev-parse "$CACHY_REMOTE/master" 2>/dev/null || true)

printf 'Fetching stock llama.cpp and CachyLLama...\n'
log "fetching $STOCK_REMOTE/master and $CACHY_REMOTE/master"

if [[ "$DRY_RUN" == "true" ]]; then
    git fetch --no-tags --dry-run "$STOCK_REMOTE" master
    git fetch --no-tags --dry-run "$CACHY_REMOTE" master
else
    git fetch --no-tags "$STOCK_REMOTE" master
    git fetch --no-tags "$CACHY_REMOTE" master
fi

STOCK_REF="$STOCK_REMOTE/master"
CACHY_REF="$CACHY_REMOTE/master"
git rev-parse --verify --quiet "$STOCK_REF^{commit}" >/dev/null || fail "$STOCK_REF does not exist"
git rev-parse --verify --quiet "$CACHY_REF^{commit}" >/dev/null || fail "$CACHY_REF does not exist"

NEW_STOCK_SHA=$(git rev-parse "$STOCK_REF")
NEW_CACHY_SHA=$(git rev-parse "$CACHY_REF")

# Count new commits since last known position
if [[ -n "$PREV_STOCK_SHA" && "$PREV_STOCK_SHA" != "$NEW_STOCK_SHA" ]]; then
    NEW_UPSTREAM_COUNT=$(git rev-list --count "$PREV_STOCK_SHA..$NEW_STOCK_SHA")
else
    NEW_UPSTREAM_COUNT=0
fi

if [[ -n "$PREV_CACHY_SHA" && "$PREV_CACHY_SHA" != "$NEW_CACHY_SHA" ]]; then
    NEW_CACHY_COUNT=$(git rev-list --count "$PREV_CACHY_SHA..$NEW_CACHY_SHA")
else
    NEW_CACHY_COUNT=0
fi

if [[ "$NEW_UPSTREAM_COUNT" == "0" && "$NEW_CACHY_COUNT" == "0" ]]; then
    log "no new commits on upstream or cachyllama since last rebase"
    printf 'Nothing new to rebase. upstream and CachyLLama unchanged since last run.\n'
    exit 0
fi

printf '\n'
log "new upstream commits since last rebase: $NEW_UPSTREAM_COUNT"
log "new cachyllama commits since last rebase: $NEW_CACHY_COUNT"
printf 'New upstream commits since last rebase: %s\n' "$NEW_UPSTREAM_COUNT"
printf 'New CachyLLama commits since last rebase: %s\n' "$NEW_CACHY_COUNT"

# Find the current base of halo
if git show-ref --verify --quiet "refs/heads/$BASE_BRANCH" &&
   git merge-base --is-ancestor "$BASE_BRANCH" "$TARGET_BRANCH"; then
    OLD_BASE=$BASE_BRANCH
elif git merge-base --is-ancestor "$CACHY_REF" "$TARGET_BRANCH"; then
    OLD_BASE=$CACHY_REF
else
    fail "$TARGET_BRANCH is not based on $BASE_BRANCH or $CACHY_REF"
fi

OLD_BASE_SHA=$(git rev-parse "$OLD_BASE")
OLD_STOCK_SHA=${PREV_STOCK_SHA:-$OLD_BASE_SHA}
OLD_CACHY_SHA=${PREV_CACHY_SHA:-$NEW_CACHY_SHA}

STAMP=$(date -u +%Y%m%d-%H%M%S)
SAFE_TARGET=${TARGET_BRANCH//\//-}
BACKUP_BRANCH="backup/$SAFE_TARGET-before-layered-rebase-$STAMP"
WORK_BRANCH="rebase-cachyllama-$STAMP"

log "backup branch: $BACKUP_BRANCH"
log "work branch: $WORK_BRANCH"
log "old stock SHA: $OLD_STOCK_SHA"
log "old cachyllama SHA: $OLD_CACHY_SHA"

printf '\n'
log "switching to work branch $WORK_BRANCH from cachyllama/master"
git branch "$BACKUP_BRANCH" "$TARGET_BRANCH"
git branch "$WORK_BRANCH" "$CACHY_REF"

printf '\nRebasing CachyLLama changes onto %s...\n' "$STOCK_REF"
log "starting CachyLLama rebase onto $STOCK_REF"
git switch --quiet "$WORK_BRANCH"
if ! git rebase --onto "$STOCK_REF" "$STOCK_REF"; then
    log "CachyLLama rebase FAILED on conflict"
    printf '\nCachyLLama rebase stopped on a conflict.\n' >&2
    printf 'Resolve it and run git rebase --continue, or run git rebase --abort.\n' >&2
    printf 'Backup branch: %s\n' "$BACKUP_BRANCH" >&2
    exit 1
fi
REBASED_CACHY=$(git rev-parse HEAD)
log "CachyLLama rebased to $REBASED_CACHY"

printf '\nRebasing %s-only changes onto the rebased CachyLLama layer...\n' "$TARGET_BRANCH"
log "starting $TARGET_BRANCH rebase onto $REBASED_CACHY"
git switch --quiet "$TARGET_BRANCH"
if ! git rebase --onto "$REBASED_CACHY" "$OLD_BASE"; then
    log "$TARGET_BRANCH rebase FAILED on conflict"
    printf '\n%s rebase stopped on a conflict.\n' "$TARGET_BRANCH" >&2
    printf 'Resolve it and run git rebase --continue, or run git rebase --abort.\n' >&2
    printf 'Rebased CachyLLama branch: %s\n' "$WORK_BRANCH" >&2
    printf 'Backup branch: %s\n' "$BACKUP_BRANCH" >&2
    exit 1
fi
REBASED_HALO=$(git rev-parse HEAD)
log "halo rebased to $REBASED_HALO"

NEW_HALO_COUNT=$(git rev-list --count "$OLD_BASE_SHA..$TARGET_BRANCH" 2>/dev/null || echo "0")
log "halo has $NEW_HALO_COUNT unique commits on top of rebased CachyLLama"

git branch --force "$BASE_BRANCH" "$REBASED_CACHY"
git branch --set-upstream-to="$BASE_BRANCH" "$TARGET_BRANCH" >/dev/null
git branch --delete "$WORK_BRANCH" >/dev/null

# Commit count from CachyLlama (rebased on latest upstream)
CACHY_COMMIT_COUNT=$(git rev-list --count "$STOCK_REF..$BASE_BRANCH")
HALO_UNIQUE_COUNT=$(git rev-list --count "$BASE_BRANCH..$TARGET_BRANCH")

log "rebase complete: BASE_BRANCH=$BASE_BRANCH=$REBASED_CACHY"
log "rebase complete: TARGET_BRANCH=$TARGET_BRANCH=$REBASED_HALO"

printf '\nLayered rebase complete:\n'
printf '  stock:       %s (%s new commits)\n' "$STOCK_REF" "$NEW_UPSTREAM_COUNT"
printf '  CachyLLama:  %s (%s commits on stock, %s new since last)\n' "$BASE_BRANCH" "$CACHY_COMMIT_COUNT" "$NEW_CACHY_COUNT"
printf '  %s:  %s unique commits (%s rebased)\n' "$TARGET_BRANCH" "$HALO_UNIQUE_COUNT" "$NEW_HALO_COUNT"
printf '  backup:      %s\n' "$BACKUP_BRANCH"

# Run build if requested
BUILD_OK=false
if [[ "$BUILD" == "true" ]]; then
    printf '\nRunning ROCm build verification...\n'
    log "starting ROCm build"
    BUILD_LOG="/tmp/rebase-build-$(date -u +%Y%m%d-%H%M%S).log"
    if scripts/build-halo.sh rocm > "$BUILD_LOG" 2>&1; then
        BUILD_OK=true
        log "ROCm build PASSED"
        printf 'ROCm build: PASSED\n'
    else
        BUILD_OK=false
        log "ROCm build FAILED (see $BUILD_LOG)"
        printf 'ROCm build: FAILED (see %s)\n' "$BUILD_LOG" >&2
        printf 'Continuing (--build failed but not blocking rebase completion)\n' >&2
    fi
fi

# Push if requested
if [[ "$PUSH" == "true" ]]; then
    printf '\nPushing %s to origin...\n' "$TARGET_BRANCH"
    log "pushing $TARGET_BRANCH to origin"
    if git push origin "$TARGET_BRANCH"; then
        log "push succeeded"
        printf 'Push succeeded.\n'
    else
        log "push FAILED"
        printf 'Push FAILED.\n' >&2
    fi
fi

# Update progress.md with change summary if requested
if [[ "$UPDATE_PROGRESS" == "true" ]]; then
    printf '\nUpdating progress.md with change summary...\n'
    log "updating progress.md"

    UPSTREAM_SINCE=$(git log --oneline "$PREV_STOCK_SHA..$STOCK_REF" 2>/dev/null | head -20 || true)
    CACHY_SINCE=$(git log --oneline "$PREV_CACHY_SHA..$CACHY_REF" 2>/dev/null | head -20 || true)

    PROGRESS_UPDATE=$(cat <<PROGRESS

## Rebase summary ($(date -u +%Y-%m-%dT%H:%M:%SZ))

- New upstream commits: $NEW_UPSTREAM_COUNT
- New CachyLLama commits: $NEW_CACHY_COUNT
- Halo unique commits: $HALO_UNIQUE_COUNT
- Build: $([ "$BUILD_OK" = "true" ] && echo "passed" || echo "not run or failed")

### New upstream commits
$(echo "$UPSTREAM_SINCE" || echo "(none)")

### New CachyLLama commits
$(echo "$CACHY_SINCE" || echo "(none)")
PROGRESS
)
    printf '%s\n' "$PROGRESS_UPDATE" >> "$ROOT_DIR/progress.md"

    git add progress.md
    git commit -m "rebase: update progress.md after weekly rebase ($(date -u +%Y-%m-%d))" --allow-empty 2>/dev/null || true
    log "progress.md updated and committed"
fi

# Update state file
if [[ "$UPDATE_STATE" == "true" ]]; then
    cat > "$STATE_FILE" <<EOF
STOCK_SHA=$NEW_STOCK_SHA
CACHY_SHA=$NEW_CACHY_SHA
BASE_SHA=$OLD_BASE_SHA
REBASED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TARGET_BRANCH=$TARGET_BRANCH
EOF
    log "state file updated at $STATE_FILE"
fi

if [[ "$UPDATE_LOG" == "true" ]]; then
    log "=== rebase session complete ==="
fi

printf '\nDone.\n'
log "=== weekly-rebase complete for $TARGET_BRANCH ==="