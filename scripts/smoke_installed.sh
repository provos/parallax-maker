#!/usr/bin/env bash
# Standalone startup smoke test for the *installed* parallax-maker package.
#
# Verifies the console script actually starts, serves the JSON API, and
# serves the built Svelte app (index.html plus one of its own hashed
# assets) -- i.e. that the frontend really is packaged into the wheel/venv,
# not just present in a source checkout. Run this after `pip install` of a
# built wheel/sdist (see .github/workflows/python-app.yml's "Smoke test the
# installed package" step), or locally against an editable install.
#
# Usage: scripts/smoke_installed.sh [port]
set -euo pipefail

PORT="${1:-8119}"
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"

if ! command -v parallax-maker >/dev/null 2>&1; then
  echo "error: 'parallax-maker' console script not found on PATH" >&2
  echo "       (install the package first, e.g. 'pip install .' or 'pip install -e .')" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

echo "Starting parallax-maker on ${BASE_URL} (cwd=${WORK_DIR})..."
(
  cd "$WORK_DIR"
  exec parallax-maker --host "$HOST" --port "$PORT"
) >"${WORK_DIR}/server.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for ${BASE_URL}/api/v1/health..."
READY=0
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null "${BASE_URL}/api/v1/health"; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "error: server process exited before becoming ready" >&2
    cat "${WORK_DIR}/server.log" >&2
    exit 1
  fi
  sleep 0.5
done
if [[ "$READY" -ne 1 ]]; then
  echo "error: server did not become ready within 30s" >&2
  cat "${WORK_DIR}/server.log" >&2
  exit 1
fi

HEALTH_BODY="$(curl -fsS "${BASE_URL}/api/v1/health")"
echo "Health: ${HEALTH_BODY}"
case "$HEALTH_BODY" in
  *'"ok": true'*|*'"ok":true'*) ;;
  *) echo "error: unexpected health response: ${HEALTH_BODY}" >&2; exit 1 ;;
esac

INDEX_HTML="$(curl -fsS "${BASE_URL}/")"
if [[ "$INDEX_HTML" != *'<div id="app">'* ]]; then
  echo "error: '/' did not return the built Svelte app's index.html" >&2
  echo "$INDEX_HTML" >&2
  exit 1
fi
echo "Fetched / (index.html, ${#INDEX_HTML} bytes)"

ASSET_PATH="$(printf '%s' "$INDEX_HTML" | grep -o '/assets/[A-Za-z0-9._-]*\.js' | head -n1)"
if [[ -z "$ASSET_PATH" ]]; then
  echo "error: could not find a hashed asset reference in index.html" >&2
  exit 1
fi
ASSET_STATUS="$(curl -fsS -o /dev/null -w '%{http_code}' "${BASE_URL}${ASSET_PATH}")"
if [[ "$ASSET_STATUS" != "200" ]]; then
  echo "error: fetching ${ASSET_PATH} returned HTTP ${ASSET_STATUS}" >&2
  exit 1
fi
echo "Fetched ${ASSET_PATH} (HTTP ${ASSET_STATUS})"

echo "OK: parallax-maker installed package smoke test passed."
