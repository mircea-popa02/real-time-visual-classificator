"""senses run | status | pause | resume | search | tail"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import ConfigError, data_dir, load_config
from .events import Event
from .timefmt import format_events

log = logging.getLogger("senses")


def _control(method, path, body=None, port=8765):
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def _events(rows):
    return [Event(row["ts"], row["source"], row["kind"], row["text"]) for row in rows]


def _capture_line(status):
    until = status.get("paused_until")
    return f"capture: {status['capture']}" + (f" until {time.strftime('%H:%M:%S', time.localtime(until))}"
                                              if until else "")


def _open_sources(cfg, log_lines):
    from .sources.audio import AudioStream, find_devices
    from .sources.camera import Camera
    from .sources.screen import SourceError, detect_screen

    screen, probe_log = detect_screen(cfg.screen)
    log_lines += [f"  screen probe: {line}" for line in probe_log if screen is None or "ok" not in line]
    camera = None
    if cfg.camera.enabled:
        camera = Camera(cfg.camera)
        try:
            camera.read()
        except SourceError as exc:
            log_lines.append(f"  camera: off ({exc})")
            camera = None
    mic = desktop = None
    if cfg.audio.enabled:
        mic_device, desktop_device = find_devices(cfg.audio)
        mic = AudioStream(mic_device, "mic") if mic_device else None
        desktop = AudioStream(desktop_device, "desktop") if desktop_device else None
    return screen, camera, (mic, desktop)


async def _run(cfg):
    from .daemon import Daemon
    from .launcher import start_server
    from .mcp_server import serve
    from .model import ModelClient
    from .processors.screen_text import RapidOcrEngine
    from .store import Store

    started = time.monotonic()
    server = await asyncio.to_thread(start_server, cfg.model, data_dir() / "llama.log")
    notes = []
    try:
        ready = time.monotonic() - started
        client = ModelClient(server.url, cfg.model.alias, workers=cfg.model.slots, timeout=cfg.model.timeout)
        screen, camera, audio = await asyncio.to_thread(_open_sources, cfg, notes)
        daemon = Daemon(cfg, Store(cfg.store.resolved_path()), client, screen, camera, audio, RapidOcrEngine())
        mcp_url = f"http://{cfg.server.host}:{cfg.server.port}/mcp"
        print(f"senses: model ready in {ready:.1f}s at {server.url}")
        for name, source in daemon.status()["sources"].items():
            print(f"  {name:<8} {source['backend'] or 'off'}")
        print("\n".join(notes) if notes else "", end="\n" if notes else "")
        print(f"MCP: {mcp_url}\n  add it to Claude Code: claude mcp add --transport http senses {mcp_url}", flush=True)
        tasks = [asyncio.create_task(daemon.run()), asyncio.create_task(serve(daemon, cfg.server.host, cfg.server.port))]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            daemon.stop()
            for task in tasks:
                task.cancel()
            if camera is not None:
                camera.close()
            client.close()
    finally:
        server.stop()


def build_parser():
    parser = argparse.ArgumentParser(prog="senses", description="Local eyes and ears for a desktop.")
    parser.add_argument("--config", type=Path, help="config.toml (default: ~/.config/senses/config.toml)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="Start the model, capture and the MCP server")
    commands.add_parser("status", help="Show presence, capture state and sources")
    pause = commands.add_parser("pause", help="Pause capture (presence still runs unless --all)")
    pause.add_argument("duration", nargs="?", help="e.g. 10m; omit to pause until resume")
    pause.add_argument("--all", action="store_true", help="Also stop the camera and presence rules")
    commands.add_parser("resume", help="Resume capture")
    search = commands.add_parser("search", help="Full-text search of the history")
    search.add_argument("query")
    search.add_argument("--since", help="10m, 2h, 1d or an ISO time")
    search.add_argument("--limit", type=int, default=20)
    tail = commands.add_parser("tail", help="Print new events as they happen")
    tail.add_argument("--since", default="1m")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"senses: {exc}", file=sys.stderr)
        return 2
    port = cfg.server.port
    if args.command == "run":
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
        try:
            asyncio.run(_run(cfg))
        except KeyboardInterrupt:
            pass
        return 0
    try:
        if args.command == "status":
            from .mcp_server import status_text
            print(status_text(_control("GET", "/control/status", port=port)))
        elif args.command == "pause":
            print(_capture_line(_control("POST", "/control/pause", {"duration": args.duration, "all": args.all},
                                         port=port)))
        elif args.command == "resume":
            print(_capture_line(_control("POST", "/control/resume", port=port)))
        elif args.command == "search":
            query = urllib.parse.urlencode({k: v for k, v in (("q", args.query), ("since", args.since),
                                                              ("limit", args.limit)) if v is not None})
            print(format_events(_events(_control("GET", f"/control/search?{query}", port=port))))
        elif args.command == "tail":
            since = args.since
            seen = set()
            while True:
                rows = _control("GET", f"/control/timeline?{urllib.parse.urlencode({'since': since})}", port=port)
                fresh = [row for row in rows if (row["ts"], row["source"], row["text"]) not in seen]
                if fresh:
                    print(format_events(_events(fresh)), flush=True)
                    seen.update((row["ts"], row["source"], row["text"]) for row in fresh)
                since = "10s"
                time.sleep(1)
    except (ConnectionRefusedError, urllib.error.URLError, ConnectionError):
        print("senses is not running (start it with: senses run)", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0
