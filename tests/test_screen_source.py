import sys
from pathlib import Path
import os
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.config import ScreenConfig
from senses.sources.screen import (CommandBackend, MssBackend, PortalScreenshotBackend, SourceError,
                                   detect_screen)

WRITE_PNG = (f'{sys.executable} -c "import cv2,numpy,sys; '
             f'cv2.imwrite(sys.argv[1], numpy.zeros((20,30,3),numpy.uint8))" {{path}}')


class DetectScreenTests(unittest.TestCase):
    def test_wayland_prefers_portal_then_known_command(self):
        backend, log = detect_screen(ScreenConfig(), env={"WAYLAND_DISPLAY": "wayland-0"},
                                     which=lambda n: "/usr/bin/spectacle" if n == "spectacle" else None, probe=False)
        self.assertIsInstance(backend, (PortalScreenshotBackend, CommandBackend))
        self.assertTrue(any("spectacle" in line or "portal" in line for line in log))

    def test_wayland_without_session_bus_uses_command(self):
        backend, _ = detect_screen(ScreenConfig(), env={"WAYLAND_DISPLAY": "wayland-0"},
                                   which=lambda n: "/usr/bin/grim" if n == "grim" else None, probe=False)
        self.assertIsInstance(backend, CommandBackend)
        self.assertEqual(backend.command, "grim {path}")

    def test_x11_uses_mss(self):
        backend, _ = detect_screen(ScreenConfig(), env={"DISPLAY": ":0"}, which=lambda n: None, probe=False)
        self.assertIsInstance(backend, MssBackend)

    def test_forced_command(self):
        backend, _ = detect_screen(ScreenConfig(backend="command", command="grim {path}"), env={},
                                   which=lambda n: None, probe=False)
        self.assertEqual(backend.command, "grim {path}")

    def test_nothing_available(self):
        backend, log = detect_screen(ScreenConfig(), env={"WAYLAND_DISPLAY": "wayland-0"},
                                     which=lambda n: None, probe=False)
        self.assertIsNone(backend)
        self.assertTrue(log)

    def test_probe_skips_failing_candidate(self):
        backend, log = detect_screen(ScreenConfig(backend="command", command="false {path}"), env={},
                                     which=lambda n: None, probe=True)
        self.assertIsNone(backend)
        self.assertTrue(any("false" in line for line in log))


class CommandBackendTests(unittest.TestCase):
    def test_reads_and_deletes_image(self):
        backend = CommandBackend(WRITE_PNG)
        frames = backend.capture()
        self.assertEqual(frames[0].name, "desktop")
        self.assertEqual(frames[0].image.shape, (20, 30, 3))
        self.assertEqual(os.listdir(backend.tmpdir), [])

    def test_failing_command_raises_source_error(self):
        with self.assertRaises(SourceError):
            CommandBackend("false {path}").capture()


if __name__ == "__main__":
    unittest.main()
