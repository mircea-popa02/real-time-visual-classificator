# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Run all tests (offline, no server or OpenCV needed): `python3 tests/test_decision.py` (same as `./start.sh --test`)
- Run one test: `python3 tests/test_decision.py DecisionTests.test_missing_candidate_fails`
- Full app (needs llama.cpp's `llama-server` and `setsid`; set `LLAMA_BIN_DIR` if llama.cpp is not on PATH): `./start.sh [run.py flags]`, e.g. `./start.sh --image photo.jpg --describe`
- Model server only: `./start.sh --server-only`
- App against an already-running server: `.venv/bin/python run.py --url http://127.0.0.1:8060/v1 [flags]` (`python3 run.py --help` for all flags)

There is no linter or formatter configured. Tests use stdlib `unittest`; the core package has no runtime dependencies (HTTP via `urllib`), and OpenCV is an optional `webcam` extra imported lazily inside `cli.webcam()`.

## Architecture

`start.sh` is the real entry point. It creates `.venv` (via `uv sync --extra webcam`, or `venv`+`pip`) and picks a free port (8060, falling back to 8061–8079 unless `VISION_PORT` is set). `VISION_MODEL` (`gemma4-e4b` default, `gemma3-4b`) and `VISION_DEVICE` (`auto` default, detected via `llama-server --list-devices`; `gpu`; `cpu`) choose the server command and its `-ngl`/`-c`/`-np` flags. Gemma 4 runs with `--reasoning-budget 0`, and with `--offline` when its GGUF is already in the HF cache. The server runs under `setsid` with logs in `.run/llama.log`. The script polls `/health` every 0.25 s, then runs `run.py`. On exit it kills the whole server **process group** (`kill -- -$pid`), which is why `setsid` is required. `run.py` just puts `src/` on `sys.path` so nothing needs installing.

Per-frame pipeline (`cli._decide` → `decision.py`), all against llama.cpp's OpenAI-compatible `/chat/completions`. Every request sends `chat_template_kwargs.enable_thinking=false` and puts the image before the text:

1. **Observe** (`local_observe`): one request constrained by a llama.cpp-specific GBNF `grammar` (`observation_grammar`) to a single line, `objects | setting | light [| sentence]`. With `--objects any` (default) names are free lowercase nouns; with `--objects all` they are limited to the catalog in `catalog.py`. Keep this output terse: generated tokens dominate latency (~0.3 s per frame on GPU). The step runs whenever an image is present and neither `--config` nor an explicit object list is given.
2. **Score** (`local_decide`, only with `--verify`, an explicit `--objects a,b`, `--scene` or `--config`): one request per question with `max_completion_tokens: 1`, `logprobs`, `top_logprobs: 128`. Options are rendered as letters `A`, `B`, … in the prompt (`prompt_for`), and `probabilities_for` softmax-normalizes the first token's logprobs over those letters. It raises `DecisionError` if an option letter is missing and could be significant. Questions run in a thread pool (`--parallel`), and an `on_answer` callback streams each result as it completes.

Question types (`binary`, `choice` with dict criteria, `score` with list criteria; 2–20 options) are defined by `options_for` and shaped into results by `answer_for`. See `examples/scene.json` for the `--config` format, which replaces the object catalog flow entirely.

In webcam mode, the main thread captures and previews frames. Only one frame is in flight at a time, on a daemon thread; frames are dropped while it is busy. Progress events (`observation`, `answer`) are emitted only when scoring follows the observation. They go through a `queue.SimpleQueue` and are printed from the main loop; otherwise the whole frame is printed once by `print_result`. All user-facing failures should raise `DecisionError`, which `main()` turns into exit code 2.

Probabilities are normalized token scores, not calibrated detector confidence. Keep that framing in user-facing text.
