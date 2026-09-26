#!/usr/bin/env bash
# From an extracted folder: ./start.sh [optional run.py flags]
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
./start.sh                 Start Gemma 3 4B and classify webcam frames
./start.sh --image FILE    Classify one image instead
./start.sh --test          Run the built-in offline tests
Other application options: python3 run.py --help
HELP
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install python3 and retry." >&2
  exit 1
fi

if [[ "${1:-}" == "--test" ]]; then
  exec python3 tests/test_decision.py
fi

if ! command -v llama >/dev/null 2>&1; then
  echo "Cannot find 'llama'. Install llama.cpp and check that 'llama serve --help' works." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  echo "Preparing webcam dependency with uv..."
  uv sync --extra webcam
else
  echo "Preparing Python virtual environment..."
  if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv || {
      echo "Could not create .venv. Install your distribution's python3-venv package." >&2
      exit 1
    }
  fi
  .venv/bin/python -m pip install 'opencv-python==5.0.0.93'
fi

python_bin="$(pwd)/.venv/bin/python"
"$python_bin" -c 'import cv2' || {
  echo "OpenCV could not load. Check the installation above." >&2
  exit 1
}

port="${VISION_PORT:-8060}"
if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo "VISION_PORT must be a port from 1024 to 65535." >&2
  exit 1
fi
if ! python3 - "$port" <<'PY'
import socket, sys
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
then
  echo "Port $port is already in use. Set VISION_PORT to another free port." >&2
  exit 1
fi

mkdir -p .run
echo "Starting Gemma 3 4B vision on CPU; first run may download model weights."
echo "Server log: $(pwd)/.run/llama.log"
llama serve --vision-gemma-4b-default --alias local-vision \
  -c 2048 -ngl 0 --host 127.0.0.1 --port "$port" > .run/llama.log 2>&1 &
server_pid=$!

cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ready=0
for ((attempt=1; attempt<=900; attempt++)); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Vision server stopped during startup. Recent log:" >&2
    tail -n 30 .run/llama.log >&2
    exit 1
  fi
  if "$python_bin" - "$port" <<'PY' >/dev/null 2>&1
import sys, urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=1) as response:
    assert response.status == 200
PY
  then
    ready=1
    break
  fi
  if (( attempt % 5 == 0 )); then
    echo "Waiting for the vision model... ($((attempt * 2)) s)"
  fi
  sleep 2
done

if (( ! ready )); then
  echo "Vision server did not become ready. See .run/llama.log" >&2
  exit 1
fi

echo "Vision model ready. Starting classifier..."
"$python_bin" run.py --url "http://127.0.0.1:$port/v1" "$@"
