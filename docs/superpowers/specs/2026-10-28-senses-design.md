# Senses: local eyes and ears for a desktop, with an MCP interface

Status: approved by the owner · 2026-10-28

## Intent

A personal background service that watches and listens while its owner sits at the computer, keeps a searchable
**text** history of what was on screen, said and heard, and exposes that context to LLM agents through MCP.
It runs 100% locally on the owner's machine. Agents (for example Claude Code) use it to "see" the screen and the
room on demand, to search what happened earlier, and to wait for something to happen without burning tokens on
every frame.

It grows out of this repository's webcam classifier, which itself started as an open-source take on JEV: a
decision API that answers structured questions (yes/no, choice, score) with token-probability scores rather than
free text. That idea stays at the core: the caller asks the questions at runtime; nothing is hard-coded.

Why it matters to the owner:

- **Personal context on demand.** "What error did I see ten minutes ago?", "What was said in yesterday's call about
  the deadline?", "Which page was I on when I found that link?"
- **Agents stop working blind.** When the owner asks "why does it crash?", the agent can read the current screen
  and the last minutes of activity instead of the owner copy-pasting.
- **Presence as an input.** Arriving at and leaving the desk become events that drive actions.
- **Watch and wake.** An agent hands over a condition ("tell me when the spinner disappears", "when an error
  dialog appears", "when the speaker in the call stops talking") and the service watches locally until it is true.

## Goals

1. Capture, while the owner is present and not paused:
   - every connected screen: text via OCR, plus a short "what is the user doing" summary from the vision model;
   - the webcam: presence (anyone sitting at the desk counts, not only the owner; no identity recognition),
     and an occasional scene note;
   - the microphone ("me") and desktop audio ("desktop"): speech transcribed to text.
2. Store **text only** in a local, searchable history with configurable retention.
3. Expose an MCP server so agents can query the present and the past, ask JEV-style scored questions about a live
   source, and block until a condition is met.
4. Run rules on events. First rule (the owner's): when the owner leaves the desk, pause media and lock the screen
   after about 10 seconds.
5. A pause control that stops all capture immediately, plus a visible indication that capture is on.
6. **Portable:** no hard-coded monitors, desktop environments, devices or models. Sources are auto-detected
   adapters; everything is overridable in a config file. Must run on another machine with a different setup.
7. Light enough to stay on all day next to normal work on a 24 GB GPU.

## Non-goals

- Storing raw video, raw audio or screenshot archives. (Transient frames and audio buffers live only in memory.)
- Multiple machines, network capture nodes, Raspberry Pi, Tailscale. (Explicitly dropped by the owner.)
- Speaker identification beyond "me" (mic) versus "desktop" (system audio).
- Video-rate object detection, tracking or bounding boxes.
- A rich GUI. A tray indicator and CLI are enough.
- Semantic or embedding search in v1 (full-text search first; the storage design should not prevent adding it).

## Anti-goals (would make it a failure even if it works)

- Sending anything off the machine without the owner explicitly asking for it. (The owner decided that
  registering the MCP server with an agent and invoking that agent is the consent: MCP results go to the agent,
  including cloud agents, without extra approval steps.)
- Capturing while the owner is away or capture is paused, except screen watches an agent registered explicitly
  (see "Watches while away").
- Noticeably slowing the computer: stuttering desktop, fans maxed, GPU memory starving the owner's other work.
- Breaking or degrading on a machine that is not this one (other DE, other monitors, X11, no webcam, no GPU):
  missing capabilities must degrade to "that source is off", never crash the service.
- Hard-coded vocabularies (the old 36-object catalog and fixed scene questions are removed).

## Constraints and measured facts

Measured on the owner's machine (RX 7900 XTX 24 GB via Vulkan, KDE Plasma on Wayland, two monitors at 1.25
scale, one portrait; PipeWire; Logitech C615 webcam with mic). These are spike results, not guarantees.

| Item | Result |
|---|---|
| Model | Gemma 4 E4B Q4_0 + mmproj via `llama-server` (llama.cpp b11933, Vulkan), reasoning off |
| VRAM | ~4.7 GB with 32k context, 4 slots, `--image-max-tokens 1120` |
| Server start | ~1.3–1.5 s warm with `--offline`; ~10 s on a cold first load |
| Webcam glance (objects, setting, light) | ~0.3 s per frame with a GBNF one-line grammar |
| Yes/no scored question (1 token, logprobs) | ~0.2 s |
| Screen summary from Gemma (1120 image tokens) | 1.6–2.5 s per monitor; tends to be verbose unless constrained |
| **Gemma as OCR on dense code** | **fails** (returned only line numbers) |
| RapidOCR (PP-OCR via onnxruntime, CPU) | excellent on code, URLs and Romanian diacritics; 0.4 s small region, 1.1–1.9 s per monitor |
| Gemma 4 audio transcription | English 11 s clip perfect in 0.2–0.6 s; Romanian 15 s mic clip good in ~0.3 s; English technical terms inside Romanian speech get misheard ("Claude Code" → "Cloud Code") |
| Full-desktop screenshot via Spectacle CLI | ~0.45 s for 4000×2560 (both monitors in one image) |
| ScreenCast portal | present (version 5, supports restore tokens) |
| GStreamer `pipewiresrc` | **not installed** here (needs `gstreamer1.0-pipewire`, requires sudo) |

Other constraints:

- Python (the existing code is Python; transcription, OCR and MCP libraries are most mature there). *Architect's call.*
- llama.cpp's OpenAI-compatible server is the default model backend; requests already in use: images first in the
  message, `chat_template_kwargs.enable_thinking=false`, GBNF `grammar` for terse outputs, `logprobs` +
  `top_logprobs` for scored answers.
- The owner speaks Romanian and English; transcription must auto-detect language.

## Architecture

One long-running process, `senses` (daemon), started at login as a user service. Inside it:

```
 sources (adapters)          processors                 core                    interfaces
 ┌──────────────┐      ┌──────────────────────┐   ┌────────────────┐   ┌───────────────────────┐
 │ screen(s)    │─────▶│ change detect → OCR  │──▶│                │   │ MCP (streamable HTTP, │
 │ camera(s)    │─────▶│ presence, glance     │──▶│  event bus  ───┼──▶│  localhost)           │
 │ mic          │─────▶│ VAD → transcribe     │──▶│  presence FSM  │   │ CLI (status, pause,   │
 │ desktop audio│─────▶│ VAD → transcribe     │──▶│  rules engine  │   │  search, tail)        │
 └──────────────┘      │ activity summary     │   │  store (SQLite)│   │ tray indicator        │
                       └──────────┬───────────┘   └────────────────┘   └───────────────────────┘
                                  │
                         model client (OpenAI-compatible: llama.cpp by default) · OCR engine (pluggable)
```

### Sources: auto-detected adapters

Each source kind has an adapter interface and several backends. At startup the daemon probes backends in order,
picks the first that works, logs what it picked and why the others were skipped, and the config can force one.
No geometry, device names or DE assumptions in code.

- **Screen.** Backends, in probe order:
  1. ScreenCast portal + PipeWire (per-monitor streams, one consent dialog, then a restore token; works on any
     Wayland DE that implements the portal). Needs a PipeWire video consumer, for example GStreamer `pipewiresrc`.
  2. Screenshot portal (any Wayland DE; whole desktop per call).
  3. `mss` (X11, Windows, macOS; per monitor).
  4. Command backend: a configured screenshot command (`spectacle -b -n -f -o {path}`, `grim {path}`,
     `screencapture -x {path}`, …), whole desktop.

  Monitor boundaries come from the backend when it has them. A whole-desktop image is acceptable: it is treated
  as one screen, and nothing may assume a layout. Change detection, not monitor knowledge, decides what to OCR.
- **Camera.** OpenCV `VideoCapture`; device chosen by config or the first that opens. Missing camera means presence
  falls back to input activity (see presence).
- **Audio.** Microphone and desktop audio (loopback or monitor) through a cross-platform capture library, for
  example `soundcard` (PulseAudio/PipeWire monitors on Linux, WASAPI loopback on Windows); macOS desktop audio
  needs a virtual device and is out of scope to automate. Devices are default-or-configured, never hard-coded.

### Processors

- **Screen text.** Capture every `screen.interval` (default 2 s). Compare to the previous frame with a cheap
  downscaled difference; only on meaningful change run OCR (RapidOCR by default; engine is pluggable). Store the
  OCR text as a snapshot, plus the line-level diff against the previous snapshot of that screen so search finds
  what *appeared*. Active window or app title is stored when an adapter can provide it (optional per platform).
- **Activity summary.** On significant screen change, at most every `summary.min_interval` (default 30 s), ask the
  vision model for one constrained line: application, what the user is doing. Grammar-limited to stay short.
- **Presence.** At ~1 Hz, ask the model one JEV-style scored question on the webcam frame ("Is a person sitting
  in front of the computer?") using the existing 1-token logprob method (~0.2 s). Feed the probability into a
  presence state machine with hysteresis (enter/leave thresholds, minimum durations) to avoid flapping.
  Fallback without camera: keyboard/mouse idle time where the platform exposes it.
- **Speech.** Voice-activity detection (for example Silero VAD) cuts mic and desktop audio into utterances
  (cap ~30 s); each utterance goes to the model as `input_audio` for transcription. Mic segments are tagged
  `me`, desktop segments `desktop`. Optional echo guard: when desktop audio and mic carry the same speech
  (speakers, no headphones), prefer the desktop tag.
- **Scene note.** Occasional (default every 60 s while present) one-line webcam glance using the existing grammar
  approach; open vocabulary.

### Core

- **Event bus.** In-process pub/sub; every processor output is an event `{ts, source, kind, text, meta}`.
- **Presence FSM.** States: `present`, `away`, `paused`. Capture of screen and audio runs only in `present`.
  The camera keeps a low-rate presence check while `away` so arrival is detected; nothing is stored while away
  except the transitions themselves.
- **Rules engine.** Rules in config: `on` event (with optional match), `after` delay that is cancelled if the
  state flips back, `run` commands or built-in actions. Example, the owner's first rule:

  ```toml
  [[rules]]
  on = "presence.away"
  after = "10s"
  run = ["playerctl -a pause", "loginctl lock-session"]
  ```

  Built-in actions are thin wrappers that pick a platform-appropriate command (media pause, lock screen, desktop
  notification) so the default config works on other systems; commands remain overridable.
- **Store.** SQLite with FTS5: `events` table (`id, ts, source, kind, text, meta JSON`) plus an FTS index on
  `text`. A retention job deletes rows older than `retention.days` (default 30). Database path under the XDG data
  dir (platform equivalent elsewhere). Text only.

### Interfaces

- **MCP server**, hosted by the daemon over streamable HTTP on localhost (so it shares state with the running
  daemon). Registered once, e.g. `claude mcp add --transport http senses http://127.0.0.1:<port>/mcp`. Tools:

  | Tool | Purpose |
  |---|---|
  | `status()` | presence, paused or not, active sources and backends, model health |
  | `now(sources?)` | the current state: per-screen OCR text and last activity summary, recent transcript lines, presence |
  | `look(source, prompt?)` | fresh capture; returns a short description of that screen or the camera |
  | `ask(source, questions)` | JEV-style: questions of type binary/choice/score answered with normalized token probabilities on a fresh capture |
  | `search(query, since?, until?, sources?, limit?)` | full-text search over history, newest first, with timestamps |
  | `timeline(since, until?, sources?)` | chronological events in a window ("what happened in the last 10 minutes") |
  | `watch(condition, source, ttl?)` | registers a condition and returns a watch id; conditions: text regex on OCR or transcript, a scored question above a threshold on a live source, or a presence change. Watches live in the daemon (default ttl 1 h) |
  | `wait_for(watch_id \| condition, timeout?)` | blocks until the watch fires or `timeout` (default 60 s, capped below the client's tool timeout), then returns either the matching event or "still waiting"; the agent calls again to keep waiting. Long waits are a series of short calls, never one long one |
  | `watches()` / `unwatch(id)` | list and cancel watches |
  | `pause(duration?)` / `resume()` | capture control |

  Returned content is text. Returning raw images to the agent is not part of v1.
- **CLI:** `senses run | status | pause [dur] | resume | search <q> | tail`.
- **Tray indicator** showing capturing / paused / away, with a pause toggle. Must degrade silently where no tray
  exists.

### Model backend

- Default: the daemon starts `llama-server` itself (the logic of today's `start.sh`: auto device, cached weights
  with `--offline`, free port, process-group cleanup), or connects to a configured URL.
- All model calls go through one client with a priority queue: presence checks and `wait_for` / `ask` calls jump
  ahead of activity summaries and transcription backlog, so interactive use stays snappy.
- Any OpenAI-compatible server should work for the basic calls; llama.cpp-only features (GBNF grammar,
  `input_audio`) are used when available and the client degrades (for example, JSON schema instead of grammar)
  when not.

### Configuration

One TOML file in the XDG config dir; every key optional; defaults are auto-detected. Sections: `model`, `screen`,
`camera`, `audio`, `presence`, `ocr`, `store`/`retention`, `rules`, `privacy` (excluded apps or window titles
whose screens are skipped, e.g. password managers, when the platform can tell).

### Existing code

- `decision.py` (scored questions, logprob normalization, grammar observation) becomes the model-facing core used
  by `ask`, presence and the scene note.
- `catalog.py` (static objects and scene questions) is removed.
- The webcam CLI demo either becomes a thin client of the daemon or is retired. *Builder's call.*
- `start.sh` launch logic moves into the daemon; the script may remain as a convenience wrapper.

### Resource budget

Targets while present, on the owner's machine: model VRAM ≤ 8 GB; average CPU of the daemon ≤ 15% of one
machine's total; no sustained fan ramp attributable to it. Means to stay inside it:

- OCR only on changed regions (bounding box of the frame diff, padded to whole text lines), not whole screens.
- A token-bucket rate limit for OCR; when the change rate stays high (video, games, fast scrolling), drop to
  periodic snapshots and mark them as such.
- Presence rate adapts: 1 Hz around transitions, 0.3 Hz when stable.
- Transcription and summaries are low priority in the model queue and may lag; interactive MCP calls never wait
  behind them.

### Watches while away

The owner allows agents to watch screens while the owner is away. While at least one screen watch is active:

- screen capture continues for that watch only; frames are evaluated in memory and only the firing event is stored;
- the away rule still pauses media, but postpones locking the screen until the last screen watch fires, is
  cancelled, or expires (watch ttl capped at 30 min while away), because a locked screen hides what is watched;
- `status()` and the indicator show that a watch is keeping the screen unlocked.

Audio and scene capture stay off while away.

### Pause levels

- `pause` (default): no screen, audio or scene capture and nothing stored. The camera presence check keeps running
  in memory only, so the away rule (lock screen) still works.
- `pause --all`: camera off as well; presence rules are suspended. The indicator shows which level is active.

### Privacy safeguards in v1

- Exclusions by app or window title are best effort: implemented where the platform exposes the active window
  (X11, GNOME via extension, KDE via a KWin script or D-Bus where feasible), silently unavailable elsewhere, and
  `status()` reports whether they are active.
- No redaction of passwords or keys in v1 (owner's decision); the pause control is the owner's tool for sensitive
  moments.

## Calls made by the architect (not decided by the owner)

- Python; one daemon process; MCP over local streamable HTTP rather than a separate stdio process.
- RapidOCR for screen text; Gemma 4 E4B for presence, summaries, scene notes and transcription (no separate Whisper).
- SQLite + FTS5; default retention 30 days.
- Presence from a 1 Hz scored webcam question with hysteresis.
- Capture only while present; the camera keeps watching at low rate while away to detect return.
- Portability scope for v1: built and tested on Linux (Wayland on any desktop that implements the portals, and X11).
  Adapter interfaces leave room for Windows and macOS, which are not built or tested in v1 (no test machine).
- No sudo in the default path: the whole-desktop screenshot backend (portal or command) is built and tuned first;
  the per-monitor ScreenCast backend is used automatically when a PipeWire video consumer is installed.
- `wait_for` is a short blocking call over daemon-side watches (see the MCP table).
- Resource budget, pause levels, watches while away and best-effort exclusions as described above.

## Left to the builder

- Exact libraries for the PipeWire consumer, audio capture and VAD, after verifying on this machine.
- Change-detection thresholds, presence thresholds and timings (start from the defaults above, tune on real use).
- Package layout, naming, test strategy.
- How much of the old CLI survives.

## Owner decisions after review

- MCP results may go to any agent the owner invokes, cloud included; no per-call approval and no redaction in v1.
- Screens may be watched while the owner is away (see "Watches while away"; the lock postponement is the
  architect's call and the owner may veto it).
- Presence means anyone at the desk.
