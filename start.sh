#!/usr/bin/env bash
# ./start.sh [senses run options]   — prepare .venv, then run the senses daemon in the foreground.
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
./start.sh                 Start senses: model server, capture, MCP on http://127.0.0.1:8765/mcp
./start.sh                 History UI: http://127.0.0.1:8765/
./start.sh --test          Run the offline tests
Other commands (while it runs): .venv/bin/python run.py status | pause [10m] [--all] | resume | search QUERY | tail
Settings: ~/.config/senses/config.toml (see examples/config.toml)
Environment: LLAMA_BIN_DIR=DIR if llama.cpp's binaries are not on PATH
HELP
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install python3 and retry." >&2
  exit 1
fi

if [[ -n "${LLAMA_BIN_DIR:-}" ]]; then
  PATH="$LLAMA_BIN_DIR:$PATH"
fi

if command -v uv >/dev/null 2>&1; then
  uv sync -q
else
  if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv || {
      echo "Could not create .venv. Install your distribution's python3-venv package." >&2
      exit 1
    }
  fi
  .venv/bin/python -m pip install -q -e . httpx
fi

if [[ "${1:-}" == "--test" ]]; then
  exec .venv/bin/python -m unittest discover -s tests
fi

if ! command -v llama-server >/dev/null 2>&1; then
  echo "Cannot find 'llama-server'. Install llama.cpp, or set LLAMA_BIN_DIR to its binaries." >&2
  exit 1
fi

exec .venv/bin/python run.py run "$@"
