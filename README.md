# Real-time visual classificator

Point your Linux laptop webcam at a scene and watch object estimates appear in the terminal. The application uses **Gemma 3 4B QAT vision** through your `llama` installation. The `4B` is the model size; the preset you showed is named `--vision-gemma-4b-default` and refers to Gemma **3**, not a model named “Gemma 4.”

## Run from the zip

1. Extract the zip and open a terminal **inside** `real-time-visual-classificator/`.
2. Check that `llama serve --help` and `python3 --version` work. The launcher uses `uv` if installed; otherwise Python's `venv` and `pip`.
3. Run:

```sh
./start.sh
```

The first run creates `.venv`, installs OpenCV, and may download Gemma model weights through `llama`. It then starts the model server, opens your webcam preview, and prints results in the same terminal. Keep internet access available for the first setup. Press **Esc** in the preview or **Ctrl-C** to stop; the launcher stops its server too. The model runs on CPU by default so it can start on a 16 GB AMD laptop. Expect seconds per analyzed frame rather than video-rate inference.

Example output:

```text
Camera open. Gemma 3 4B will examine frames; Esc or Ctrl-C to stop.
[2026-09-26T13:42:08] Gemma vision | 5.2s
  Vision shortlist: person, laptop, chair
  Jev-style: Is person visible? 91% yes
  Jev-style: Is laptop visible? 83% yes
  Jev-style: Is chair visible? 46% yes
  Visible: person 91%, laptop 83%
```

The model proposes up to eight names from a catalog of 36 everyday objects and then asks a yes/no question for each. The percentages come from normalized model token scores. They are **not calibrated detector confidence**, and objects missed in the first shortlist cannot be recovered by the second step. The image stays on your laptop; the local vision path does not use your JEV API key.

## Useful commands

| Command | Effect |
| --- | --- |
| `./start.sh` | Start server and webcam; print the live shortlist, questions, and likely objects. |
| `./start.sh --image photo.jpg` | Analyze one image instead of the webcam. |
| `./start.sh --max-edge 384 --interval 2` | Send smaller frames and wait at least two seconds between requests. |
| `./start.sh --no-preview` | Print results without a preview window; Ctrl-C stops it. |
| `./start.sh --test` | Run offline tests without starting the server. |
| `./start.sh --help` | Show the launcher commands. |

If your camera is not device 0, add `--camera 1`. A custom server port can be selected with `JEV_VISION_PORT=8061 ./start.sh`. Startup messages are in the terminal; detailed server messages go to `.run/llama.log`. If setup fails, read the terminal error and that log.

## Your JEV access

The webcam uses local **Jev-style interrogations** (short, lettered yes/no questions scored from Gemma token probabilities). These are not calls to the hosted JEV API. JEV's [documented Decisions endpoint](https://www.jevai.org/docs) accepts text/JSON state and questions, but does not document webcam images. For an actual JEV API text decision, set your key and run:

```sh
export JEV_API_KEY='your-key'
./start.sh --backend jev --state '{"inventory":["laptop","chair"]}' --objects laptop,chair
```

This text command skips the camera and local server. Never put your key in a file you commit or share.

## Code structure

| Path | Purpose |
| --- | --- |
| `start.sh` | Single Linux entry point: install OpenCV, launch/wait for `llama serve`, run the app, clean up. |
| `run.py` | Python entry point that imports directly from `src/`; no package installation needed. |
| `src/jev_vision/catalog.py` | The 36 object names and optional scene questions. |
| `src/jev_vision/decision.py` | Local vision shortlist, lettered logprob questions, probability math, and JEV text API client. |
| `src/jev_vision/cli.py` | Camera capture, frame scheduling, terminal output, arguments, and image loading. |
| `src/jev_vision/__main__.py` | Optional module entry point. |
| `tests/test_decision.py` | Offline request, scoring, and launcher behavior tests. |
| `examples/scene.json` | Example of custom Jev-style questions. |
| `pyproject.toml` | Optional package metadata and webcam dependency. |

The script uses the webcam in one worker at a time and skips older frames while Gemma is busy. It cannot promise real-time video speed on every 16 GB laptop. No model weights or personal credentials are inside the zip.

Inspired by Allan Riordan Boll's September 25, 2026 article, “A Jev-like wrapper for LLMs, including vision models,” supplied with this project. This implementation is not affiliated with JEV or the article's author.
