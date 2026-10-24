#!/usr/bin/env bash
# Start the pinned local Pokémon Showdown server in the background.
# Usage: scripts/start_showdown.sh [port]
# Env: SHOWDOWN_DIR (default: ../pokemon-showdown next to this repo), SHOWDOWN_LOG.
set -euo pipefail

PINNED_COMMIT="c0845f33d381402dfcb18ce324c20a94f18af075"
PORT="${1:-8000}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOWDOWN_DIR="${SHOWDOWN_DIR:-$(dirname "$REPO_DIR")/pokemon-showdown}"
SHOWDOWN_LOG="${SHOWDOWN_LOG:-$(dirname "$REPO_DIR")/showdown.log}"

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
# shellcheck disable=SC1091
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

if ! command -v node >/dev/null || [[ "$(command -v node)" == /mnt/* ]]; then
    echo "error: native Linux node not found (install Node 22 via nvm, see docs/SETUP.md)" >&2
    exit 1
fi
if [ ! -d "$SHOWDOWN_DIR" ]; then
    echo "error: $SHOWDOWN_DIR not found (see docs/SETUP.md)" >&2
    exit 1
fi

HEAD_COMMIT="$(git -C "$SHOWDOWN_DIR" rev-parse HEAD)"
if [ "$HEAD_COMMIT" != "$PINNED_COMMIT" ]; then
    echo "warning: $SHOWDOWN_DIR is at $HEAD_COMMIT, pinned commit is $PINNED_COMMIT" >&2
fi

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT already in use; assuming Showdown is running"
    exit 0
fi

cd "$SHOWDOWN_DIR"
[ -f config/config.js ] || cp config/config-example.js config/config.js
nohup node pokemon-showdown start --no-security "$PORT" >"$SHOWDOWN_LOG" 2>&1 &
echo "started Showdown (pid $!) on port $PORT, log: $SHOWDOWN_LOG"

for _ in $(seq 1 60); do
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        echo "listening on port $PORT"
        exit 0
    fi
    sleep 1
done
echo "error: Showdown did not start listening within 60s; see $SHOWDOWN_LOG" >&2
exit 1
