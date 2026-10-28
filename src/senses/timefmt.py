"""Time parsing and the shared one-line-per-event text format used by the CLI and MCP tools."""

from datetime import datetime
import time

from .config import ConfigError, parse_duration

SOURCE_WIDTH = 14
_INDENT = 8 + 2 + SOURCE_WIDTH + 2


def parse_since(value, now):
    """A timestamp from a relative duration ("10m", "2h", "1d") or an ISO 8601 local time."""
    try:
        return now - parse_duration(value)
    except ConfigError:
        pass
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        raise ValueError(f"Invalid time {value!r}: use a duration like 10m, 2h, 1d "
                         "or an ISO time like 2026-10-28T18:00") from None


def _text(event):
    if event.kind.startswith("presence."):
        return event.kind.split(".", 1)[1]
    return event.text


def format_events(events, now=None):
    """Lines of `HH:MM:SS  source  text`; older days get a date prefix; continuation lines are indented."""
    if not events:
        return "(nothing)"
    today = datetime.fromtimestamp(time.time() if now is None else now).date()
    lines = []
    for event in events:
        moment = datetime.fromtimestamp(event.ts)
        stamp = moment.strftime("%H:%M:%S")
        if moment.date() != today:
            stamp = moment.strftime("%Y-%m-%d ") + stamp
        indent = " " * (_INDENT + len(stamp) - 8)
        first, *rest = (_text(event) or "").split("\n")
        lines.append(f"{stamp}  {event.source:<{SOURCE_WIDTH}}  {first}".rstrip())
        lines.extend(indent + line for line in rest)
    return "\n".join(lines)
