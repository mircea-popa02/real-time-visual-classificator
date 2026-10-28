"""MCP tools over the daemon, plus small JSON control routes for the CLI. Everything binds to localhost."""

from datetime import datetime
import functools
import time

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import ConfigError, parse_duration
from .decision import DecisionError
from .timefmt import format_events, parse_since
from .watches import parse_condition

INSTRUCTIONS = ("Senses keeps a local text history of the owner's screens, speech and presence. "
                "Use now/look/ask for the present, search/timeline for the past, and watch + wait_for "
                "to wait for something to appear without polling the screen yourself.")
MAX_WAIT = 90


def _clock(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _errors_as_text(fn):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except KeyError as exc:
            return f"error: unknown {exc.args[0] if exc.args else 'item'}"
        except (ValueError, ConfigError, DecisionError, OSError) as exc:
            return f"error: {exc}"
    return wrapper


def status_text(status):
    lines = [f"presence: {status['presence']}", f"capture: {status['capture']}"]
    if status["paused_until"]:
        lines.append(f"paused until: {_clock(status['paused_until'])}")
    lines.append("sources:")
    for name, source in status["sources"].items():
        detail = source["backend"] or "-"
        if source["error"]:
            detail += f" ({source['error']})"
        lines.append(f"  {name}: {source['state']} {detail}")
    lines.append(f"model: {status['model']['url']} ({'healthy' if status['model']['healthy'] else 'unreachable'})")
    lines.append(f"watches: {status['watches']}" + (" (keeping the screen unlocked)" if status["lock_held_by_watch"] else ""))
    lines.append(f"exclusions: {status['exclusions']}")
    return "\n".join(lines)


def _answers_text(result):
    lines = []
    for name, answer in result["answers"].items():
        if answer["type"] == "binary":
            lines.append(f"{name}: {answer['probability']:.0%} yes")
        elif answer["type"] == "choice":
            lines.append(f"{name}: {answer['choice']} (p={answer['probabilities'][answer['choice']]:.0%})")
        else:
            lines.append(f"{name}: score {answer['score']:.2f}")
    return "\n".join(lines)


def _event_json(event):
    return {"ts": event.ts, "source": event.source, "kind": event.kind, "text": event.text}


def build_server(daemon):
    server = MCPServer("senses", instructions=INSTRUCTIONS)
    tool = lambda fn: server.tool()(_errors_as_text(fn))

    @tool
    async def status() -> str:
        """Presence, capture state, sources and their backends, model health, active watches."""
        return status_text(daemon.status())

    @tool
    async def now(sources: list[str] | None = None) -> str:
        """What is on screen right now (OCR text and last activity summary per screen), recent speech, camera scene."""
        state = daemon.now(sources)
        parts = [f"presence: {state['presence']}"]
        for name, screen in state["screens"].items():
            parts.append(f"## screen:{name}" + (f"\n{screen['summary']}" if screen["summary"] else "")
                         + f"\n{screen['text'] or '(no text)'}")
        parts.append("## speech (last 5 min)\n" + format_events(state["recent_speech"]))
        if state["scene"]:
            parts.append(f"## camera\n{state['scene']}")
        return "\n\n".join(parts)

    @tool
    async def look(source: str, prompt: str | None = None) -> str:
        """Fresh capture of 'camera', 'screen' or 'screen:<name>'; with a prompt, the model answers it about the image."""
        return await daemon.look(source, prompt)

    @tool
    async def ask(source: str, questions: dict) -> str:
        """JEV-style scored questions on a fresh capture. questions: {name: {"type": "binary"|"choice"|"score",
        "instructions": str, "criteria": ...}}; answers are normalized token probabilities, not calibrated."""
        return _answers_text(await daemon.ask(source, questions))

    @tool
    async def search(query: str, since: str | None = None, until: str | None = None,
                     sources: list[str] | None = None, limit: int = 20) -> str:
        """Full-text search of the history (screens, speech, presence), newest first. since/until: 10m, 2h, 1d or ISO time."""
        now_ts = time.time()
        return format_events(daemon.store.search(
            query, since=parse_since(since, now_ts) if since else None,
            until=parse_since(until, now_ts) if until else None, sources=sources, limit=limit))

    @tool
    async def timeline(since: str = "10m", until: str | None = None, sources: list[str] | None = None) -> str:
        """Everything that happened in a time window, oldest first. since/until: 10m, 2h, 1d or ISO time."""
        now_ts = time.time()
        return format_events(daemon.store.timeline(
            parse_since(since, now_ts), until=parse_since(until, now_ts) if until else None, sources=sources))

    @tool
    async def watch(condition: dict, ttl: str = "1h") -> str:
        """Register a condition to watch locally: {"text": "<regex>", "sources": [...]},
        {"question": "<yes/no>", "source": "camera"|"screen", "threshold": 0.8} or {"presence": "present"|"away"}.
        Then call wait_for with the returned id."""
        registered = daemon.watches.add(parse_condition(condition), ttl=parse_duration(ttl))
        return f"watch {registered.id} registered, expires {_clock(registered.expires)}"

    @tool
    async def wait_for(watch_id: str, timeout: float = 60) -> str:
        """Wait up to `timeout` seconds (max 90) for a watch to fire; call again while it says 'still waiting'."""
        fired = await daemon.watches.wait(watch_id, min(MAX_WAIT, max(1, timeout)))
        if fired is None:
            return f"still waiting (watch {watch_id})"
        return "fired: " + format_events([fired.fired])

    @tool
    async def watches() -> str:
        """Active and fired-but-unread watches."""
        rows = [f"{w.id}  {w.condition.summary()}  expires {_clock(w.expires)}  {'fired' if w.fired else 'active'}"
                for w in daemon.watches.list()]
        return "\n".join(rows) or "(no watches)"

    @tool
    async def unwatch(watch_id: str) -> str:
        """Cancel a watch."""
        return "removed" if daemon.watches.remove(watch_id) else "error: unknown watch"

    @tool
    async def pause(duration: str | None = None, all: bool = False) -> str:
        """Pause capture (presence still checked so auto-lock works); all=true also stops the camera and rules."""
        daemon.pause(parse_duration(duration) if duration else None, all=all)
        until = f" until {_clock(daemon.paused_until)}" if daemon.paused_until else ""
        return f"paused {'everything' if all else 'capture'}{until}"

    @tool
    async def resume() -> str:
        """Resume capture."""
        daemon.resume()
        return "capture resumed"

    @server.custom_route("/control/status", methods=["GET"])
    async def control_status(request: Request):
        return JSONResponse(daemon.status())

    @server.custom_route("/control/pause", methods=["POST"])
    async def control_pause(request: Request):
        body = await request.json() if await request.body() else {}
        try:
            duration = parse_duration(body["duration"]) if body.get("duration") else None
        except ConfigError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        daemon.pause(duration, all=bool(body.get("all")))
        return JSONResponse(daemon.status())

    @server.custom_route("/control/resume", methods=["POST"])
    async def control_resume(request: Request):
        daemon.resume()
        return JSONResponse(daemon.status())

    @server.custom_route("/control/search", methods=["GET"])
    async def control_search(request: Request):
        params = request.query_params
        try:
            since = parse_since(params["since"], time.time()) if params.get("since") else None
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        events = daemon.store.search(params.get("q", ""), since=since, limit=int(params.get("limit", 20)))
        return JSONResponse([_event_json(e) for e in events])

    @server.custom_route("/control/timeline", methods=["GET"])
    async def control_timeline(request: Request):
        try:
            since = parse_since(request.query_params.get("since", "1m"), time.time())
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse([_event_json(e) for e in daemon.store.timeline(since)])

    return server


async def serve(daemon, host, port):
    await build_server(daemon).run_streamable_http_async(host=host, port=port)
