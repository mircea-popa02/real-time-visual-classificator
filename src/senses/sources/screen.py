"""Screen capture backends, probed at startup. Nothing here assumes a desktop, monitor layout or device."""

import concurrent.futures
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import uuid

import cv2
import numpy as np


class SourceError(Exception):
    """A source could not produce data right now."""


@dataclass
class Frame:
    name: str
    image: np.ndarray
    ts: float


KNOWN_COMMANDS = [
    ("spectacle", "spectacle -b -n -f -o {path}"),
    ("grim", "grim {path}"),
    ("gnome-screenshot", "gnome-screenshot -f {path}"),
    ("screencapture", "screencapture -x {path}"),
]


class MssBackend:
    name = "mss"

    def __init__(self):
        self._mss = None

    def capture(self):
        try:
            import mss
            if self._mss is None:
                self._mss = mss.mss()
            frames = []
            for index, monitor in enumerate(self._mss.monitors[1:], start=1):
                shot = np.asarray(self._mss.grab(monitor))
                frames.append(Frame(f"monitor-{index}", cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR), time.time()))
        except Exception as exc:
            raise SourceError(f"mss capture failed: {exc}") from exc
        if not frames:
            raise SourceError("mss found no monitors")
        return frames


class CommandBackend:
    def __init__(self, command):
        self.command = command
        self.name = f"command:{shlex.split(command)[0]}"
        self.tmpdir = tempfile.mkdtemp(prefix="senses-")
        os.chmod(self.tmpdir, 0o700)

    def capture(self):
        path = Path(self.tmpdir) / "screen.png"
        argv = [part.replace("{path}", str(path)) for part in shlex.split(self.command)]
        try:
            result = subprocess.run(argv, capture_output=True, timeout=10)
            if result.returncode != 0:
                raise SourceError(f"{argv[0]} exited with {result.returncode}: "
                                  f"{result.stderr.decode(errors='replace').strip()[:200]}")
            image = cv2.imread(str(path))
            if image is None:
                raise SourceError(f"{argv[0]} produced no readable image")
            return [Frame("desktop", image, time.time())]
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SourceError(f"{argv[0]} failed: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)


class PortalScreenshotBackend:
    """org.freedesktop.portal.Screenshot, non-interactive; works on Wayland desktops that implement it."""

    name = "portal"

    def __init__(self):
        self._conn = None

    def capture(self):
        try:
            from jeepney import DBusAddress, MatchRule, new_method_call
            from jeepney.bus_messages import message_bus
            from jeepney.io.blocking import open_dbus_connection
            if self._conn is None:
                self._conn = open_dbus_connection(bus="SESSION")
            token = "senses" + uuid.uuid4().hex
            sender = self._conn.unique_name[1:].replace(".", "_")
            rule = MatchRule(type="signal", interface="org.freedesktop.portal.Request", member="Response",
                             path=f"/org/freedesktop/portal/desktop/request/{sender}/{token}")
            portal = DBusAddress("/org/freedesktop/portal/desktop", bus_name="org.freedesktop.portal.Desktop",
                                 interface="org.freedesktop.portal.Screenshot")
            with self._conn.filter(rule) as queue:
                self._conn.send_and_get_reply(message_bus.AddMatch(rule))
                self._conn.send_and_get_reply(new_method_call(
                    portal, "Screenshot", "sa{sv}",
                    ("", {"handle_token": ("s", token), "interactive": ("b", False)})))
                signal = self._conn.recv_until_filtered(queue, timeout=5)
        except Exception as exc:
            raise SourceError(f"screenshot portal failed: {exc}") from exc
        response, results = signal.body
        if response != 0 or "uri" not in results:
            raise SourceError(f"screenshot portal refused (response {response})")
        path = Path(urllib.parse.unquote(urllib.parse.urlparse(results["uri"][1]).path))
        try:
            image = cv2.imread(str(path))
        finally:
            path.unlink(missing_ok=True)
        if image is None:
            raise SourceError("screenshot portal returned an unreadable file")
        return [Frame("desktop", image, time.time())]


def _has_session_bus(env):
    if env.get("DBUS_SESSION_BUS_ADDRESS"):
        return True
    runtime = env.get("XDG_RUNTIME_DIR")
    return bool(runtime) and os.path.exists(os.path.join(runtime, "bus"))


def _mss_importable():
    try:
        import mss  # noqa: F401
        return True
    except ImportError:
        return False


def _candidates(cfg, env, which, log):
    """Yield (label, factory) in probe order; unavailable ones are logged and skipped."""
    if cfg.backend == "mss":
        yield "mss", MssBackend
        return
    if cfg.backend == "portal":
        yield "portal", PortalScreenshotBackend
        return
    if cfg.backend == "command":
        if not cfg.command:
            log.append("command: skipped (screen.command is not set)")
            return
        yield cfg.command, lambda: CommandBackend(cfg.command)
        return

    def commands():
        for program, command in KNOWN_COMMANDS:
            if which(program):
                yield program, lambda command=command: CommandBackend(command)
            else:
                log.append(f"{program}: skipped (not installed)")

    if env.get("WAYLAND_DISPLAY"):
        try:
            from .screencast import ScreenCastBackend, screencast_available
        except ImportError:
            screencast_available = None
        if screencast_available and screencast_available():
            yield "screencast", ScreenCastBackend
        else:
            log.append("screencast: skipped (no pipewiresrc)")
        if _has_session_bus(env):
            yield "portal", PortalScreenshotBackend
        else:
            log.append("portal: skipped (no session bus)")
        yield from commands()
    else:
        if _mss_importable():
            yield "mss", MssBackend
        else:
            log.append("mss: skipped (not installed)")
        yield from commands()


def _probe(backend, timeout=5.0):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        frames = pool.submit(backend.capture).result(timeout=timeout)
    finally:
        pool.shutdown(wait=False)
    if not frames or not all(frame.image.size for frame in frames):
        raise SourceError("empty capture")


def detect_screen(cfg, env=os.environ, which=shutil.which, probe=True):
    log = []
    for label, factory in _candidates(cfg, env, which, log):
        try:
            backend = factory()
            if probe:
                _probe(backend)
        except (SourceError, concurrent.futures.TimeoutError, OSError) as exc:
            log.append(f"{label}: skipped ({exc or 'timed out'})")
            continue
        log.append(f"{label}: ok")
        return backend, log
    return None, log
