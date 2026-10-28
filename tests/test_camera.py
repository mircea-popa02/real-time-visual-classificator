import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.config import CameraConfig
from senses.sources.camera import Camera
from senses.sources.screen import SourceError


class FakeCapture:
    def __init__(self, index, works):
        self.index, self.works = index, works

    def isOpened(self):
        return self.works

    def set(self, *args):
        pass

    def read(self):
        return (True, np.zeros((4, 4, 3), np.uint8)) if self.works else (False, None)

    def release(self):
        pass


class CameraTests(unittest.TestCase):
    def test_picks_first_device_that_opens(self):
        opened = []
        cam = Camera(CameraConfig(), opener=lambda i: opened.append(i) or FakeCapture(i, i == 2))
        frame = cam.read()
        self.assertEqual((frame.name, frame.image.shape), ("camera", (4, 4, 3)))
        self.assertEqual(opened, [0, 1, 2])

    def test_no_camera_raises(self):
        cam = Camera(CameraConfig(), opener=lambda i: FakeCapture(i, False))
        with self.assertRaises(SourceError):
            cam.read()


if __name__ == "__main__":
    unittest.main()
