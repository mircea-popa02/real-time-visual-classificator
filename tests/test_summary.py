import concurrent.futures
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.processors.summary import Throttle, activity_grammar, frame_uri, scene_note, summarize
from senses.sources.screen import Frame


def done_future(value):
    future = concurrent.futures.Future()
    future.set_result(value)
    return future


class SummaryTests(unittest.TestCase):
    def test_grammar_shape(self):
        grammar = activity_grammar()
        self.assertTrue(grammar.startswith("root ::= [A-Z]"))
        self.assertIn('": "', grammar)
        self.assertIn("*", grammar)  # markdown emphasis is excluded

    def test_throttle(self):
        t = Throttle(30.0)
        self.assertTrue(t.ready(0.0))
        self.assertFalse(t.ready(10.0))
        self.assertTrue(t.ready(31.0))

    def test_summarize_builds_event(self):
        seen = {}

        def describe(images, prompt, priority, **kw):
            seen.update(kw)
            return done_future("Konsole: running unit tests")

        ev = summarize(SimpleNamespace(describe=describe),
                       Frame("desktop", np.zeros((50, 50, 3), np.uint8), 7.0)).result(1)
        self.assertEqual((ev.source, ev.kind, ev.text), ("screen:desktop", "screen.summary", "Konsole: running unit tests"))
        self.assertEqual(seen["grammar"], activity_grammar())

    def test_scene_note_formats_observation(self):
        observation = {"objects": ["man", "chair"], "setting": "indoors", "lighting": "bright",
                       "summary": "A man sits at a desk."}
        client = SimpleNamespace(observe=lambda images, priority, **kw: done_future(observation))
        ev = scene_note(client, Frame("camera", np.zeros((50, 50, 3), np.uint8), 3.0)).result(1)
        self.assertEqual((ev.source, ev.kind), ("camera", "camera.scene"))
        self.assertEqual(ev.text, "A man sits at a desk. (man, chair; indoors, bright)")

    def test_frame_uri_downscales(self):
        uri = frame_uri(Frame("d", np.zeros((3000, 4000, 3), np.uint8), 0.0), max_edge=1600)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
