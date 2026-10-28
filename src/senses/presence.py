"""Presence: a hysteresis state machine over per-reading probabilities, plus an input-idle fallback."""

from enum import Enum
import shutil
import subprocess


class PresenceState(str, Enum):
    PRESENT = "present"
    AWAY = "away"


class PresenceFSM:
    def __init__(self, cfg, initial=PresenceState.AWAY):
        self.cfg = cfg
        self.state = initial
        self._pending_since = None
        self._last_transition = float("-inf")

    def update(self, probability, now):
        """Feed one reading; returns the new state on a transition. None (no reading) never changes state."""
        if probability is None:
            self._pending_since = None
            return None
        if self.state == PresenceState.AWAY:
            qualifies, needed, target = probability >= self.cfg.enter, self.cfg.enter_after, PresenceState.PRESENT
        else:
            qualifies, needed, target = probability <= self.cfg.leave, self.cfg.leave_after, PresenceState.AWAY
        if not qualifies:
            self._pending_since = None
            return None
        if self._pending_since is None:
            self._pending_since = now
        if now - self._pending_since < needed:
            return None
        self.state = target
        self._pending_since = None
        self._last_transition = now
        return target

    def interval(self, now):
        if self._pending_since is not None or now - self._last_transition < 30:
            return 1 / self.cfg.fast_hz
        return 1 / self.cfg.slow_hz


def _dbus_idle_ms():
    from jeepney import DBusAddress, new_method_call
    from jeepney.io.blocking import open_dbus_connection
    address = DBusAddress("/org/freedesktop/ScreenSaver", bus_name="org.freedesktop.ScreenSaver",
                          interface="org.freedesktop.ScreenSaver")
    with open_dbus_connection(bus="SESSION") as conn:
        reply = conn.send_and_get_reply(new_method_call(address, "GetSessionIdleTime"), timeout=1)
    # Some desktops (KDE on Wayland) answer with an error message instead of a number.
    if not reply.body or not isinstance(reply.body[0], int):
        raise RuntimeError(f"GetSessionIdleTime unavailable: {reply.body!r}")
    return reply.body[0]


def idle_seconds():
    """Seconds since the last keyboard or mouse input, or None when the platform does not tell."""
    try:
        return _dbus_idle_ms() / 1000.0
    except Exception:
        pass
    if shutil.which("xprintidle"):
        try:
            return int(subprocess.run(["xprintidle"], capture_output=True, text=True, timeout=1).stdout) / 1000.0
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None


def idle_probability(idle, threshold=60.0):
    if idle is None:
        return None
    return 1.0 if idle < threshold else 0.0
