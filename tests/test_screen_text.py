import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.processors.screen_text import OcrLine, ScreenTextProcessor, changed_boxes, changed_region, new_lines
from senses.sources.screen import Frame


class FakeOcr:
    def __init__(self, responses=None, repeat=None):
        self.responses, self.repeat, self.calls, self.shapes = list(responses or []), repeat, 0, []

    def read(self, image):
        self.calls += 1
        self.shapes.append(image.shape)
        return self.responses.pop(0) if self.responses else list(self.repeat or [])


class ChangedRegionTests(unittest.TestCase):
    def test_identical_is_none_and_first_is_whole(self):
        img = np.zeros((400, 800, 3), np.uint8)
        self.assertIsNone(changed_region(img, img.copy()))
        self.assertEqual(changed_region(None, img), (0, 0, 800, 400))

    def test_rectangle_bbox(self):
        a = np.zeros((400, 800, 3), np.uint8)
        b = a.copy()
        b[100:200, 300:500] = 255
        x, y, w, h = changed_region(a, b)
        self.assertTrue(296 <= x <= 300 and 96 <= y <= 100 and 196 <= w <= 208 and 96 <= h <= 108)

    def test_tiny_change_ignored(self):
        a = np.zeros((1440, 2560, 3), np.uint8)
        b = a.copy()
        b[5:15, 2500:2540] = 255
        self.assertIsNone(changed_region(a, b, min_change=0.002))

    def test_shape_change_is_whole(self):
        self.assertEqual(changed_region(np.zeros((10, 10, 3), np.uint8), np.zeros((20, 30, 3), np.uint8)),
                         (0, 0, 30, 20))


class ChangedBoxesTests(unittest.TestCase):
    def test_separate_changes_give_separate_boxes(self):
        a = np.zeros((1200, 4000, 3), np.uint8)
        b = a.copy()
        b[100:130, 200:600] = 255      # left monitor, near the top
        b[1000:1030, 3000:3400] = 255  # right monitor, near the bottom
        boxes = sorted(changed_boxes(a, b))
        self.assertEqual(len(boxes), 2)
        (x0, y0, w0, h0), (x1, y1, w1, h1) = boxes
        self.assertLess(y0 + h0, 400)
        self.assertGreater(y1, 800)
        self.assertLess(w0 * h0 + w1 * h1, 0.1 * 4000 * 1200)

    def test_box_is_widened_around_a_typed_character(self):
        a = np.zeros((800, 2560, 3), np.uint8)
        b = a.copy()
        b[400:420, 1200:1212] = 255
        (x, y, w, h), = changed_boxes(a, b, min_change=0.0)
        self.assertLessEqual(x, 1200 - 300)
        self.assertGreaterEqual(x + w, 1212 + 300)

    def test_none_below_threshold_and_whole_on_first(self):
        img = np.zeros((100, 100, 3), np.uint8)
        self.assertEqual(changed_boxes(img, img.copy()), [])
        self.assertEqual(changed_boxes(None, img), [(0, 0, 100, 100)])


class NewLinesTests(unittest.TestCase):
    def test_counts_duplicates_and_keeps_order(self):
        self.assertEqual(new_lines(["a", "b"], ["b", "c", "a", "a"]), ["c", "a"])

    def test_similar_reread_is_not_new(self):
        old = ["self.responses, self.repeat, self.calls = list(responses or [])"]
        noisy = ["self.responses, self.repeat, self.calts = list(responses or [])"]
        self.assertEqual(new_lines(old, noisy, similarity=0.9), [])
        self.assertEqual(new_lines(old, ["Deployment complete"], similarity=0.9), ["Deployment complete"])


class ProcessorTests(unittest.TestCase):
    def test_first_frame_emits_all_then_only_new_lines(self):
        ocr = FakeOcr([[OcrLine("hello", 0, 10), OcrLine("world", 0, 300)], [OcrLine("build ok", 0, 0)]])
        p = ScreenTextProcessor(ocr, 0.002, 30)
        a = np.zeros((400, 800, 3), np.uint8)
        b = a.copy()
        b[290:320, 0:400] = 255
        first = p.process(Frame("desktop", a, 1.0))
        self.assertEqual((first.source, first.kind, first.text), ("screen:desktop", "screen.text", "hello\nworld"))
        ev = p.process(Frame("desktop", b, 2.0))
        self.assertEqual(ev.text, "build ok")
        self.assertEqual(ocr.shapes[1][1], 800)          # band spans the full width
        self.assertLess(ocr.shapes[1][0], 400)           # but not the full height
        self.assertEqual(p.lines("desktop"), ["hello", "build ok"])
        self.assertIsNone(p.process(Frame("desktop", b, 3.0)))

    def test_no_new_text_emits_nothing(self):
        ocr = FakeOcr(repeat=[OcrLine("same", 0, 0)])
        p = ScreenTextProcessor(ocr, 0.0, 30)
        p.process(Frame("d", np.zeros((50, 50, 3), np.uint8), 1.0))
        self.assertIsNone(p.process(Frame("d", np.full((50, 50, 3), 200, np.uint8), 2.0)))

    def test_rate_limit_skips_ocr_and_marks_busy(self):
        clock = [0.0]
        ocr = FakeOcr(repeat=[OcrLine("x", 0, 0)])
        p = ScreenTextProcessor(ocr, 0.0, ocr_per_minute=1, clock=lambda: clock[0])
        for i in range(3):
            p.process(Frame("d", np.full((100, 100, 3), i * 80, np.uint8), float(i)))
        self.assertEqual(ocr.calls, 1)
        clock[0] = 61.0
        ocr.repeat = [OcrLine("y", 0, 0)]
        ev = p.process(Frame("d", np.full((100, 100, 3), 250, np.uint8), 4.0))
        self.assertEqual(ocr.calls, 2)
        self.assertTrue(ev.meta["busy"])

    def test_screens_are_independent(self):
        ocr = FakeOcr(repeat=[OcrLine("t", 0, 0)])
        p = ScreenTextProcessor(ocr, 0.002, 30)
        img = np.zeros((50, 50, 3), np.uint8)
        self.assertIsNotNone(p.process(Frame("monitor-1", img, 1.0)))
        self.assertIsNotNone(p.process(Frame("monitor-2", img, 1.0)))
        self.assertEqual(p.lines("monitor-3"), [])


if __name__ == "__main__":
    unittest.main()
