"""Hardware and model fakes shared by daemon and MCP tests."""

import concurrent.futures
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.decision import DecisionError
from senses.processors.screen_text import OcrLine
from senses.sources.screen import Frame, SourceError


def done(value=None, error=None):
    future = concurrent.futures.Future()
    future.set_exception(error) if error else future.set_result(value)
    return future


def screen_frame(value, name="desktop"):
    """A frame whose pixel value FakeOcr maps to text."""
    return Frame(name, np.full((120, 160, 3), value, np.uint8), 0.0)


class FakeScreen:
    name = "fake-screen"

    def __init__(self):
        self.value, self.failures, self.captures = 0, 0, 0

    def capture(self):
        self.captures += 1
        if self.failures:
            self.failures -= 1
            raise SourceError("screen gone")
        return [screen_frame(self.value)]


class FakeCamera:
    name = "fake-camera"

    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        return Frame("camera", np.zeros((40, 40, 3), np.uint8), 0.0)

    def close(self):
        pass


class FakeOcr:
    def __init__(self, texts=None):
        self.texts = texts or {}
        self.calls = 0

    def read(self, image):
        self.calls += 1
        text = self.texts.get(int(image[0, 0, 0]))
        return [OcrLine(text, 0, 0)] if text else []


class FakeClient:
    """`presence` is the probability returned for the presence question; `down` makes every call fail."""

    def __init__(self):
        self.presence, self.scores, self.down, self.asked = 0.9, {}, False, []

    def ask(self, images, questions, state, priority):
        self.asked.append((list(questions), priority))
        if self.down:
            return done(error=DecisionError("model down"))
        answers = {name: {"type": "binary", "probability": self.scores.get(name, self.presence)}
                   for name in questions}
        return done({"answers": answers})

    def describe(self, images, prompt, priority, **kw):
        return done(error=DecisionError("model down")) if self.down else done("Konsole: testing")

    def observe(self, images, priority, **kw):
        return done({"objects": ["man"], "setting": "indoors", "lighting": "bright", "summary": "A man sits."})

    def transcribe(self, wav, priority):
        return done("salut")

    def healthy(self):
        return not self.down
