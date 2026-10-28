"""Rules: run commands some time after an event, unless the situation flips back first."""

from dataclasses import dataclass, field
import logging
import re
import shlex
import shutil
import subprocess
import threading

from .events import Event

log = logging.getLogger(__name__)

BUILTIN_ACTIONS = {
    "@media.pause": [["playerctl", "-a", "pause"]],
    "@screen.lock": [["loginctl", "lock-session"], ["xdg-screensaver", "lock"]],
    "@notify": [["notify-send", "Senses", "{text}"]],
}


def _session_bus():
    from jeepney.io.blocking import open_dbus_connection
    return open_dbus_connection(bus="SESSION")


def _mpris_players(conn):
    from jeepney.bus_messages import message_bus
    names = conn.send_and_get_reply(message_bus.ListNames(), timeout=2).body[0]
    return [name for name in names if name.startswith("org.mpris.MediaPlayer2.")]


def _mpris_send(conn, name):
    from jeepney import DBusAddress, new_method_call
    player = DBusAddress("/org/mpris/MediaPlayer2", bus_name=name, interface="org.mpris.MediaPlayer2.Player")
    conn.send_and_get_reply(new_method_call(player, "Pause"), timeout=2)


def mpris_pause():
    """Pause every MPRIS media player on the session bus; returns how many were asked."""
    with _session_bus() as conn:
        players = _mpris_players(conn)
        for name in players:
            try:
                _mpris_send(conn, name)
            except Exception as exc:
                log.warning("Could not pause %s: %s", name, exc)
    return len(players)


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
        if action == "@media.pause":
            # No playerctl: talk MPRIS over D-Bus directly.
            return lambda: mpris_pause()
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
        if callable(argv):
            if self._run is run_detached:
                threading.Thread(target=argv, daemon=True).start()
            else:
                argv()
            text = "mpris pause"
        else:
            self._run(argv)
            text = " ".join(argv)
        return Event(now, "system", "rule.ran", text, {"on": pending.rule.on, "action": action})

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
