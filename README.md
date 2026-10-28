# Real-time visual classificator

Point your Linux webcam at a scene and watch descriptions stream into the terminal. The application runs **Gemma 4 E4B** (reasoning off) through `llama-server` from llama.cpp, on the GPU when llama.cpp sees one. On an RX 7900 XTX (Vulkan) it names the visible objects in about 0.3 s per frame, roughly 3 frames per second.

## Run from GitHub

1. Clone the repository with `git clone https://github.com/mircea-popa02/real-time-visual-classificator.git` and enter it with `cd real-time-visual-classificator` (or run `git pull` if you already have it).
2. Check that `llama-server --help` and `python3 --version` work. If the llama.cpp binaries are not on `PATH`, set `LLAMA_BIN_DIR` to their folder. The launcher uses `uv` if installed; otherwise Python's `venv` and `pip`.
3. Run:

```sh
./start.sh
```

The first run creates `.venv`, installs OpenCV, and downloads the model weights (about 5 GB) through llama.cpp. Later starts use the cached weights offline, and the server is ready in 1–2 s on a warm machine (the very first load can take ~10 s). Press **Ctrl-C** to stop; the launcher stops the server and any child processes it started. A camera window is off by default, avoiding OpenCV's Qt/Wayland warnings. Add `--preview` if you want one.

Example output:

```text
Vision model ready in 1.5s. Starting classifier...
Camera open. The vision model will examine frames. Ctrl-C to stop.
[2026-10-28T08:57:50] Gemma vision | 0.3s
  Seen: man, chair, table, wall, shirt, floor | indoors, bright light
[2026-10-28T08:57:51] Gemma vision | 0.3s
  Seen: man, chair, table, wall, shirt, floor | indoors, bright light
```

By default each frame is one short request: the model names up to six visible objects (any noun, not a fixed list), the setting, and the light level. `--describe` adds a one-sentence description (~0.35 s). `--verify` adds a yes/no check per object and prints token-probability scores; it is several times slower. Those percentages come from normalized model token scores. They are **not calibrated detector confidence**, and objects missed in the first step cannot be recovered by the checks.

## Useful commands

| Command | Effect |
| --- | --- |
| `./start.sh` | Start server and webcam; print visible objects, setting and light per frame. |
| `./start.sh --describe` | Also print a one-sentence scene description. |
| `./start.sh --verify` | Also score each named object with a yes/no probability check (slower). |
| `./start.sh --objects all` | Name objects only from the built-in 36-object catalog. |
| `./start.sh --image photo.jpg` | Analyze one image instead of the webcam. |
| `./start.sh --interval 2` | Wait at least two seconds between frames (default: none). |
| `./start.sh --preview` | Also show a camera window; Esc closes it. |
| `./start.sh --server-only` | Run only the model server, for other clients. |
| `VISION_DEVICE=cpu ./start.sh` | Force CPU (`auto`, the default, uses a GPU when llama.cpp sees one). |
| `VISION_MODEL=gemma3-4b ./start.sh` | Use the older Gemma 3 4B preset via `llama serve`. |
| `./start.sh --test` | Run offline tests without starting the server. |
| `./start.sh --help` | Show the launcher commands and environment variables. |

If your camera is not device 0, add `--camera 1`. The launcher uses port 8060 and automatically tries 8061–8079 if it is busy. You can also set `VISION_PORT=8061 ./start.sh`; an explicitly selected busy port produces an error. If an older version left a server on 8060, check the listener with `ss -ltnp '( sport = :8060 )'` and stop **that PID** with `kill PID` if it belongs to your previous run. On a machine with little RAM or VRAM, stop an old model server before starting a second copy. Startup messages are in the terminal; detailed server messages go to `.run/llama.log`. If setup fails, read the terminal error and that log. On Wayland, `QT_QPA_PLATFORM=xcb ./start.sh --preview` may work if Xwayland is installed; the default terminal-only mode does not use Qt.

## Browsing the history

History lives in SQLite at `~/.local/share/senses/senses.db` (or `$XDG_DATA_HOME/senses/senses.db`;
override with `[store] path` in `~/.config/senses/config.toml`). It stores text only — OCR text, transcripts
and presence events — in one `events` table with an FTS5 index. Browse it with
[Datasette](https://datasette.io/) (read-only, any schema):

```sh
./start.sh --view
# SQLite viewer: http://127.0.0.1:8001/  (Ctrl-C to stop)
```

`--view` runs `uvx datasette <db>` when `uv` is installed, otherwise `datasette` from `PATH`
(`pipx install datasette`). The daemon does not need to be running. For a desktop GUI, open the same file
with DB Browser for SQLite. The data is also available over MCP and from the terminal
(`.venv/bin/python run.py search QUERY`, `tail`, `status`).

## How it works

Only the local vision model is used. Each frame goes to llama.cpp's OpenAI-compatible endpoint with a GBNF grammar that forces one terse line, `objects | setting | light [| sentence]`. A short line instead of JSON keeps generation to ~15 tokens, which dominates latency. Reasoning is disabled so the model answers directly. With `--verify`, the app also asks short binary questions per object and reads the first output token's log probabilities for the `A` and `B` options. No hosted API or key is involved.

## Code structure

| Path | Purpose |
| --- | --- |
| `start.sh` | Single Linux entry point: install OpenCV, launch/wait for `llama-server`, run the app, stop its server process group. |
| `run.py` | Python entry point that imports directly from `src/`; no package installation needed. |
| `src/visual_classifier/catalog.py` | The 36 object names and optional scene questions. |
| `src/visual_classifier/decision.py` | Grammar-constrained observation, parallel logprob questions, and probability math. |
| `src/visual_classifier/cli.py` | Camera capture, frame scheduling, terminal output, arguments, and image loading. |
| `src/visual_classifier/__main__.py` | Optional module entry point. |
| `tests/test_decision.py` | Offline request, scoring, and launcher behavior tests. |
| `examples/scene.json` | Example of custom binary, choice, and score questions. |
| `pyproject.toml` | Optional package metadata and webcam dependency. |

The script uses the webcam in one worker at a time and skips older frames while Gemma is busy. On CPU-only machines expect seconds per frame rather than several frames per second. No model weights or personal credentials are in the repository.

Inspired by the local vision token-probability experiment in Allan Riordan Boll's September 25, 2026 article supplied with this project. This implementation is not affiliated with the author.
