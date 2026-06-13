#!/usr/bin/env bash
#
# One-shot launcher for the NYC Rental Listing Scores app.
#
# Sets up the environment if needed (creates a virtualenv, installs
# dependencies), then starts the web UI and opens it in your browser. Scraping,
# backfill, and scoring are all driven from the "⚙ Pipeline" page inside the UI.
#
# Usage:
#   ./setup.sh                 # set up + launch on http://127.0.0.1:5000
#   PORT=8000 ./setup.sh       # override the port
#   PROVIDER=gemini ./setup.sh # override the AI-search provider
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
PORT="${PORT:-5000}"
PROVIDER="${PROVIDER:-anthropic}"

info() { printf '\033[36m%s\033[0m\n' "$1"; }
warn() { printf '\033[33m%s\033[0m\n' "$1"; }
ok()   { printf '\033[32m%s\033[0m\n' "$1"; }

# ── environment setup ─────────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { warn "python3 not found. Install Python 3 first (e.g. 'brew install python')."; exit 1; }

if [ ! -d "$VENV" ]; then
  info "Creating virtualenv in $VENV ..."
  python3 -m venv "$VENV"
fi

# Re-install only when a core dependency is missing (fast no-op otherwise).
if ! "$PY" -c "import anthropic, flask, sklearn, pandas, curl_cffi, dotenv" >/dev/null 2>&1; then
  info "Installing dependencies from requirements.txt ..."
  "$PIP" install --upgrade pip -q
  "$PIP" install -r requirements.txt -q
fi
ok "Environment ready ($("$PY" --version 2>&1))."

if [ ! -f .env ]; then
  warn "No .env file found. Add an API key to enable scoring & AI search:"
  echo "    ANTHROPIC_API_KEY=sk-ant-...   (for the anthropic provider)"
  echo "    GOOGLE_API_KEY=...             (for the gemini provider)"
fi

# Prefer whichever provider actually has a key available, unless overridden.
has_key() { grep -q "^$1=" .env 2>/dev/null || [ -n "${!1:-}" ]; }
if [ "$PROVIDER" = "anthropic" ] && ! has_key ANTHROPIC_API_KEY && has_key GOOGLE_API_KEY; then
  PROVIDER="gemini"
fi

# ── launch ────────────────────────────────────────────────────────────────────
URL="http://127.0.0.1:${PORT}"
info "Opening $URL — use the ⚙ Pipeline page to scrape, backfill, and score."

# Open the browser shortly after the server starts (best-effort, non-fatal).
(
  sleep 2
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  fi
) >/dev/null 2>&1 &

exec "$PY" web.py --provider "$PROVIDER" --port "$PORT"
