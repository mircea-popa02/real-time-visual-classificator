"""Rules: run commands some time after an event, unless the situation flips back first."""

from dataclasses import dataclass, field
import logging
import re
import shlex
import shutil
import subprocess

from .events import Event

log = logging.getLogger(__name__)

BUILTIN_ACTIONS = {
    "@media.pause": [["playerctl", "-a", "pause"]],
    "@screen.lock": [["loginctl", "lock-session"], ["xdg-screensaver", "lock"]],
    "@notify": [["notify-send", "Senses", "{text}"]],
}

_OPPOSITE = {"presence.away": "presence.present", "presence.present": "presence.away"}
_warned = set()


def _fill(argv, event):
    values = {"text": event.text, "source": event.source, "kind": event.kind}
    # Substitute after splitting, so event text can never add or split arguments.
    return [part.format_map(values) if "{" in part else part for part in argv]


def resolve_action(action, event, which=shutil.which):
    if action.startswith("@"):
        for candidate in BUILTIN_ACTIONS.get(action, []):
            if which(candidate[0]):
                return _fill(candidate, event)
        if action not in _warned:
            _warned.add(action)
            log.warning("No program available for %s", action)
        return None
    return _fill(shlex.split(action), event)


def run_detached(argv):
    try:
        subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError as exc:
        log.warning("Rule command %s failed: %s", argv[0], exc)


@dataclass
class _Pending:
    rule: object
    event: Event
    due: float
    actions: list = field(default_factory=list)


class RulesEngine:
    def __init__(self, rules, run=run_detached, which=shutil.which):
        self.rules = rules
        self._run = run
        self._which = which
        self._pending = []
        self._held = set()
        self._suspended = False

    @property
    def suspended(self):
        return self._suspended

    @suspended.setter
    def suspended(self, value):
        self._suspended = value
        if value:
            self._pending = [p for p in self._pending if not p.rule.on.startswith("presence.")]

    def on_event(self, event, now):
        if event.kind.startswith("presence.") and self._suspended:
            return
        if event.kind in _OPPOSITE:
            self._pending = [p for p in self._pending if p.rule.on != _OPPOSITE[event.kind]]
        if event.kind == "capture.paused" and event.meta.get("all"):
            self._pending = [p for p in self._pending if not p.rule.on.startswith("presence.")]
        for rule in self.rules:
            if rule.on == event.kind and (rule.match is None or re.search(rule.match, event.text)):
                self._pending.append(_Pending(rule, event, now + rule.after, list(rule.run)))

    def _execute(self, action, pending, now):
        argv = resolve_action(action, pending.event, self._which)
        if argv is None:
            return None
        self._run(argv)
        return Event(now, "system", "rule.ran", " ".join(argv), {"on": pending.rule.on, "action": action})

    def tick(self, now):
        ran = []
        for pending in [p for p in self._pending if p.due <= now]:
            waiting = []
            for action in pending.actions:
                if action in self._held:
                    waiting.append(action)
                elif (event := self._execute(action, pending, now)) is not None:
                    ran.append(event)
            pending.actions = waiting
        self._pending = [p for p in self._pending if p.actions]
        return ran

    def hold(self, action):
        self._held.add(action)

    def release(self, action, now):
        self._held.discard(action)
        return self.tick(now)
