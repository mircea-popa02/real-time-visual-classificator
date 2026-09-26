# Real-time visual classificator

Point your Linux laptop webcam at a scene and watch object estimates appear in the terminal. The application uses **Gemma 3 4B QAT vision** through your `llama` installation. The `4B` is the model size; the preset you showed is named `--vision-gemma-4b-default` and refers to Gemma **3**, not a model named “Gemma 4.”

## Run from GitHub

1. Clone the repository with `git clone https://github.com/mircea-popa02/real-time-visual-classificator.git` and enter it with `cd real-time-visual-classificator` (or run `git pull` if you already have it).
2. Check that `llama serve --help` and `python3 --version` work. The launcher uses `uv` if installed; otherwise Python's `venv` and `pip`.
3. Run:

```sh
./start.sh
```

The first run creates `.venv`, installs OpenCV, and may download Gemma model weights through `llama`. It then starts the model server, reads your webcam, and prints progress in the **same terminal**. Keep internet access available for the first setup. Press **Ctrl-C** to stop; the launcher stops the server and any child processes it started. A camera window is off by default, avoiding OpenCV's Qt/Wayland warnings. Add `--preview` if you want one. The model runs on CPU by default so it can start on a 16 GB AMD laptop. Expect tens of seconds per analyzed frame rather than video-rate inference.

Example output:

```text
Camera open. Gemma 3 4B will examine frames. Ctrl-C to stop.
[2026-09-26T13:42:08] Frame 1: asking Gemma...
  Gemma is still processing this frame...
  Vision shortlist: person, laptop, chair
  Scene: A person at a desk with a laptop. | indoors, bright light
  Model score: Is person visible? 91% yes
  Model score: Is laptop visible? 83% yes
  Model score: Is chair visible? 46% yes
[2026-09-26T13:43:15] Gemma vision | 67.0s
  Visible: person 91%, laptop 83%
```

One Gemma request produces structured JSON: a short scene summary, indoor/outdoor setting, lighting, and up to four likely objects from a catalog of 36. The yes/no checks then run **two at a time**. A line appears as soon as each stage finishes, with a heartbeat while a request is still running. Tune `--max-candidates 1..8` and `--parallel 1..4` for your CPU. The percentages come from normalized model token scores. They are **not calibrated detector confidence**, and objects missed in the first shortlist cannot be recovered by the second step.

## Useful commands

| Command | Effect |
| --- | --- |
| `./start.sh` | Start server and webcam; print the live shortlist, questions, and likely objects. |
| `./start.sh --image photo.jpg` | Analyze one image instead of the webcam. |
| `./start.sh --max-edge 384 --interval 2` | Send smaller frames and wait at least two seconds between requests. |
| `./start.sh --preview` | Also show a camera window; Esc closes it. |
| `./start.sh --max-candidates 2` | Make fewer scoring requests per frame on a slow CPU. |
| `./start.sh --parallel 1` | Run one local score request at a time if RAM or CPU contention is high. |
| `./start.sh --test` | Run offline tests without starting the server. |
| `./start.sh --help` | Show the launcher commands. |

If your camera is not device 0, add `--camera 1`. The launcher uses port 8060 and automatically tries 8061–8079 if it is busy. You can also set `VISION_PORT=8061 ./start.sh`; an explicitly selected busy port produces an error. If an older version left a server on 8060, check the listener with `ss -ltnp '( sport = :8060 )'` and stop **that PID** with `kill PID` if it belongs to your previous run. On a 16 GB laptop, stop an old model server before starting a second copy. Startup messages are in the terminal; detailed server messages go to `.run/llama.log`. If setup fails, read the terminal error and that log. On Wayland, `QT_QPA_PLATFORM=xcb ./start.sh --preview` may work if Xwayland is installed; the default terminal-only mode does not use Qt.

## How it works

Only the local vision model is used. Gemma emits structured scene JSON, then answers short binary questions about the proposed objects. The app reads the first output token's log probabilities for the `A` and `B` options and normalizes those two scores. Two checks run concurrently by default. No hosted decision API or key is involved.

## Code structure

| Path | Purpose |
| --- | --- |
| `start.sh` | Single Linux entry point: install OpenCV, launch/wait for `llama serve`, run the app, stop its server process group. |
| `run.py` | Python entry point that imports directly from `src/`; no package installation needed. |
| `src/visual_classifier/catalog.py` | The 36 object names and optional scene questions. |
| `src/visual_classifier/decision.py` | Structured Gemma observation, parallel logprob questions, and probability math. |
| `src/visual_classifier/cli.py` | Camera capture, frame scheduling, terminal output, arguments, and image loading. |
| `src/visual_classifier/__main__.py` | Optional module entry point. |
| `tests/test_decision.py` | Offline request, scoring, and launcher behavior tests. |
| `examples/scene.json` | Example of custom binary, choice, and score questions. |
| `pyproject.toml` | Optional package metadata and webcam dependency. |

The script uses the webcam in one worker at a time and skips older frames while Gemma is busy. It cannot promise real-time video speed on every 16 GB laptop. No model weights or personal credentials are in the repository.

Inspired by the local vision token-probability experiment in Allan Riordan Boll's September 25, 2026 article supplied with this project. This implementation is not affiliated with the author.
