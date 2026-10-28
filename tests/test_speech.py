import concurrent.futures
import io
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.config import AudioConfig
from senses.processors.speech import Segmenter, SpeechProcessor, mostly_latin, to_wav
from senses.sources.audio import BLOCK, find_devices


def blocks(pattern):
    for speech, n in pattern:
        for _ in range(n):
            yield np.full(BLOCK, 0.5 if speech else 0.0, np.float32)


def done_future(value):
    future = concurrent.futures.Future()
    future.set_result(value)
    return future


class SegmenterTests(unittest.TestCase):
    vad = staticmethod(lambda b: 1.0 if b[0] > 0.1 else 0.0)

    def test_utterance_ends_after_silence(self):
        seg, out = Segmenter(self.vad), []
        for b in blocks([(False, 10), (True, 40), (False, 30)]):
            out += seg.feed(b)
        self.assertEqual(len(out), 1)
        self.assertGreaterEqual(len(out[0]), 40 * BLOCK)

    def test_max_length_cut_and_short_dropped(self):
        seg, out = Segmenter(self.vad, max_length=1.0), []
        for b in blocks([(True, 100)]):
            out += seg.feed(b)
        self.assertGreaterEqual(len(out), 3)
        seg2, out2 = Segmenter(self.vad), []
        for b in blocks([(True, 3), (False, 40)]):
            out2 += seg2.feed(b)
        self.assertEqual(out2, [])


class WavTests(unittest.TestCase):
    def test_header(self):
        data = to_wav(np.zeros(16000, np.float32))
        with wave.open(io.BytesIO(data)) as w:
            self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()),
                             (16000, 1, 2, 16000))


class SpeechProcessorTests(unittest.TestCase):
    def test_publishes_tagged_transcript(self):
        published = []
        client = SimpleNamespace(transcribe=lambda wav, priority, languages=None: done_future("salut"))
        SpeechProcessor(client, published.append, "mic").handle(np.zeros(8000, np.float32), 5.0)
        self.assertEqual((published[0].source, published[0].kind, published[0].text), ("mic", "speech.me", "salut"))

    def test_desktop_tag_and_empty_dropped(self):
        published = []
        SpeechProcessor(SimpleNamespace(transcribe=lambda wav, priority, languages=None: done_future("hi")), published.append,
                        "desktop").handle(np.zeros(8000, np.float32), 1.0)
        SpeechProcessor(SimpleNamespace(transcribe=lambda wav, priority, languages=None: done_future("  ")), published.append,
                        "mic").handle(np.zeros(8000, np.float32), 2.0)
        self.assertEqual([e.kind for e in published], ["speech.desktop"])

    def test_transcription_error_is_dropped(self):
        failed = concurrent.futures.Future()
        failed.set_exception(RuntimeError("model down"))
        published = []
        SpeechProcessor(SimpleNamespace(transcribe=lambda wav, priority, languages=None: failed), published.append,
                        "mic").handle(np.zeros(8000, np.float32), 1.0)
        self.assertEqual(published, [])

    def test_languages_are_forwarded(self):
        seen = []
        client = SimpleNamespace(
            transcribe=lambda wav, priority, languages=None: seen.append(languages) or done_future("buna"))
        SpeechProcessor(client, lambda event: None, "mic", ["Romanian", "English"]).handle(
            np.zeros(8000, np.float32), 1.0)
        self.assertEqual(seen, [["Romanian", "English"]])

    def test_non_latin_hallucination_is_dropped(self):
        published = []
        SpeechProcessor(SimpleNamespace(
            transcribe=lambda wav, priority, languages=None: done_future("Я поставил принтер Epson")),
            published.append, "mic").handle(np.zeros(8000, np.float32), 1.0)
        self.assertEqual(published, [])

    def test_mostly_latin_keeps_diacritics(self):
        self.assertTrue(mostly_latin("Mă întreb dacă merge transcrierea."))
        self.assertTrue(mostly_latin("hello world"))
        self.assertFalse(mostly_latin("Я поставил принтер"))
        self.assertTrue(mostly_latin("123"))


class FindDevicesTests(unittest.TestCase):
    def test_defaults_and_loopback(self):
        calls = []
        sc = SimpleNamespace(
            default_microphone=lambda: "mic0",
            default_speaker=lambda: SimpleNamespace(name="K3"),
            get_microphone=lambda name, include_loopback=False: calls.append((name, include_loopback)) or f"loop:{name}")
        self.assertEqual(find_devices(AudioConfig(), sc=sc), ("mic0", "loop:K3"))
        self.assertEqual(calls, [("K3", True)])

    def test_missing_devices_are_none(self):
        def boom(*a, **k):
            raise RuntimeError("no device")
        sc = SimpleNamespace(default_microphone=boom, default_speaker=boom, get_microphone=boom)
        self.assertEqual(find_devices(AudioConfig(), sc=sc), (None, None))


if __name__ == "__main__":
    unittest.main()
