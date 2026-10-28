#!/usr/bin/env bash
# From an extracted folder: ./start.sh [optional run.py flags]
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
./start.sh                 Start the vision model and describe webcam frames
./start.sh --image FILE    Describe one image instead
./start.sh --server-only   Only run the model server (for other clients)
./start.sh --test          Run the built-in offline tests
Other application options: python3 run.py --help

Environment:
  VISION_MODEL=gemma4-e4b|gemma3-4b   model (default: gemma4-e4b)
  VISION_DEVICE=auto|gpu|cpu          default auto: GPU when llama.cpp sees one
  VISION_PORT=8060                    fixed port (default: first free of 8060-8079)
  LLAMA_BIN_DIR=DIR                   llama.cpp binaries not on PATH
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

server_only=0
if [[ "${1:-}" == "--server-only" ]]; then
  server_only=1
  shift
fi
if [[ -n "${LLAMA_BIN_DIR:-}" ]]; then
  PATH="$LLAMA_BIN_DIR:$PATH"
fi

hf_cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
case "${VISION_MODEL:-gemma4-e4b}" in
  gemma3-4b)
    model_label="Gemma 3 4B"
    server_cmd=(llama serve --vision-gemma-4b-default) ;;
  gemma4-e4b)
    # Reasoning off: the classifier scores the first answer token.
    model_label="Gemma 4 E4B"
    server_cmd=(llama-server -hf ggml-org/gemma-4-E4B-it-GGUF:Q4_0 --jinja --reasoning-budget 0)
    # Skip the online freshness check when the weights are already cached; it also allows running offline.
    if compgen -G "$hf_cache/models--ggml-org--gemma-4-E4B-it-GGUF/snapshots/*/gemma-4-E4B-it-Q4_0.gguf" >/dev/null; then
      server_cmd+=(--offline)
    fi ;;
  *)
    echo "VISION_MODEL must be gemma3-4b or gemma4-e4b." >&2
    exit 1 ;;
esac
if ! command -v "${server_cmd[0]}" >/dev/null 2>&1; then
  echo "Cannot find '${server_cmd[0]}'. Install llama.cpp, or set LLAMA_BIN_DIR to its binaries." >&2
  exit 1
fi
device="${VISION_DEVICE:-auto}"
if [[ "$device" == auto ]]; then
  device=cpu
  if command -v llama-server >/dev/null 2>&1 \
      && llama-server --list-devices 2>/dev/null | grep -qE '^ *(Vulkan|CUDA|ROCm|HIP|Metal|SYCL|MUSA)[0-9]*:'; then
    device=gpu
  fi
fi
case "$device" in
  cpu) device_args=(-ngl 0 -c 4096 -np 2) ;;
  gpu) device_args=(-ngl 99 -fa on -c 16384 -np 4) ;;
  *)
    echo "VISION_DEVICE must be auto, cpu or gpu." >&2
    exit 1 ;;
esac
if ! command -v setsid >/dev/null 2>&1; then
  echo "Cannot find 'setsid' (util-linux); it is needed to stop the model server cleanly." >&2
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
port_free() {
  python3 - "$1" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}
if ! port_free "$port"; then
  if [[ -n "${VISION_PORT:-}" ]]; then
    echo "Port $port is already in use. Choose a free VISION_PORT or stop the process using it." >&2
    exit 1
  fi
  for candidate in {8061..8079}; do
    if port_free "$candidate"; then
      echo "Port 8060 is in use; using port $candidate. An older server may still be running on 8060."
      port="$candidate"
      break
    fi
  done
  if [[ "$port" == 8060 ]]; then
    echo "No free port found from 8060 to 8079. Stop the old server or set VISION_PORT." >&2
    exit 1
  fi
fi

mkdir -p .run
echo "Starting $model_label vision on $device; first run may download model weights."
echo "Server log: $(pwd)/.run/llama.log"
setsid "${server_cmd[@]}" --alias local-vision \
  "${device_args[@]}" --host 127.0.0.1 --port "$port" > .run/llama.log 2>&1 &
server_pid=$!

cleanup() {
  trap - EXIT INT TERM
  # The separate session includes wrapper scripts and any children they launch.
  kill -TERM -- "-$server_pid" 2>/dev/null || true
  for ((attempt=0; attempt<20; attempt++)); do
    if ! kill -0 -- "-$server_pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  kill -KILL -- "-$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
started=$EPOCHREALTIME
for ((attempt=1; attempt<=7200; attempt++)); do
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
  if (( attempt % 40 == 0 )); then
    echo "Waiting for the vision model... ($((attempt / 4)) s)"
  fi
  sleep 0.25
done

if (( ! ready )); then
  echo "Vision server did not become ready. See .run/llama.log" >&2
  exit 1
fi

startup=$(python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$started" "$EPOCHREALTIME")
if (( server_only )); then
  echo "Vision model ready in ${startup}s at http://127.0.0.1:$port/v1 (model alias: local-vision). Ctrl-C to stop."
  wait "$server_pid"
  exit $?
fi
echo "Vision model ready in ${startup}s. Starting classifier..."
"$python_bin" run.py --url "http://127.0.0.1:$port/v1" "$@"
