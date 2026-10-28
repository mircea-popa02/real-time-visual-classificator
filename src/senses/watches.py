"""Watches: conditions an agent registers so the daemon can watch locally and wake it when they hold."""

import asyncio
from dataclasses import dataclass, field
import re
import time
from typing import Literal
import uuid

_TEXT_KINDS = ("screen.text", "speech.me", "speech.desktop")
_SHAPES = ('{"text": "<regex>", "sources": [...]}, {"question": "<yes/no question>", "source": "camera" | "screen"}, '
           'or {"presence": "present" | "away"}')


def source_matches(source, filters):
    return any(source == wanted or source.startswith(wanted + ":") for wanted in filters)


@dataclass
class Condition:
    kind: Literal["text", "question", "presence"]
    sources: list[str]
    pattern: str | None = None
    question: str | None = None
    threshold: float = 0.8
    state: str | None = None

    def summary(self):
        if self.kind == "text":
            return f"text /{self.pattern}/ on {', '.join(self.sources)}"
        if self.kind == "question":
            return f"{self.question!r} ≥ {self.threshold:.0%} on {', '.join(self.sources)}"
        return f"presence {self.state}"


def parse_condition(spec):
    if not isinstance(spec, dict):
        raise ValueError(f"A condition is an object: {_SHAPES}")
    if "text" in spec:
        try:
            re.compile(spec["text"], re.IGNORECASE)
        except (re.error, TypeError) as exc:
            raise ValueError(f"invalid regex: {exc}") from None
        return Condition("text", list(spec.get("sources") or ["screen", "mic", "desktop"]), pattern=spec["text"])
    if "question" in spec:
        source = spec.get("source", "screen")
        if source != "camera" and not source_matches(source, ["screen"]):
            raise ValueError("question watches need source 'camera', 'screen' or 'screen:<name>'")
        return Condition("question", [source], question=str(spec["question"]),
                         threshold=float(spec.get("threshold", 0.8)))
    if "presence" in spec:
        if spec["presence"] not in ("present", "away"):
            raise ValueError("presence must be 'present' or 'away'")
        return Condition("presence", ["presence"], state=spec["presence"])
    raise ValueError(f"Unknown condition; use {_SHAPES}")


@dataclass(eq=False)
class Watch:
    id: str
    condition: Condition
    created: float
    expires: float
    fired: object = None
    _signal: asyncio.Event | None = field(default=None, repr=False)

    def _set(self):
        if self._signal is not None:
            self._signal.set()


class WatchRegistry:
    def __init__(self, clock=time.time):
        self.clock = clock
        self._watches = {}

    def add(self, condition, ttl=3600.0):
        now = self.clock()
        watch = Watch(uuid.uuid4().hex[:8], condition, now, now + ttl)
        self._watches[watch.id] = watch
        return watch

    def remove(self, id):
        return self._watches.pop(id, None) is not None

    def list(self):
        return list(self._watches.values())

    def _fire(self, watch, event):
        watch.fired = event
        watch._set()

    def on_event(self, event):
        fired = []
        for watch in self._watches.values():
            if watch.fired is not None:
                continue
            condition = watch.condition
            if condition.kind == "text":
                hit = (event.kind in _TEXT_KINDS and source_matches(event.source, condition.sources)
                       and re.search(condition.pattern, event.text, re.IGNORECASE))
            elif condition.kind == "presence":
                hit = event.kind == f"presence.{condition.state}"
            else:
                hit = False
            if hit:
                self._fire(watch, event)
                fired.append(watch)
        return fired

    def question_watches(self):
        return [w for w in self._watches.values() if w.condition.kind == "question" and w.fired is None]

    def report_score(self, id, probability, event):
        watch = self._watches.get(id)
        if watch is None or watch.fired is not None or probability < watch.condition.threshold:
            return False
        self._fire(watch, event)
        return True

    def expire(self, now):
        gone = [w for w in self._watches.values() if w.fired is None and w.expires <= now]
        for watch in gone:
            del self._watches[watch.id]
            watch._set()
        return gone

    def cap_ttl(self, max_remaining, now):
        for watch in self._watches.values():
            if watch.fired is None:
                watch.expires = min(watch.expires, now + max_remaining)

    def has_screen_watch(self):
        return any(w.fired is None and source_matches(source, ["screen"])
                   for w in self._watches.values() for source in w.condition.sources)

    async def wait(self, id, timeout):
        """The watch once it fired (then it is forgotten), or None on timeout."""
        watch = self._watches[id]
        if watch.fired is None:
            if watch._signal is None:
                watch._signal = asyncio.Event()
            try:
                await asyncio.wait_for(watch._signal.wait(), timeout)
            except asyncio.TimeoutError:
                return None
        if watch.fired is None:
            return None
        self._watches.pop(id, None)
        return watch
