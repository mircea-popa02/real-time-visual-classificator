"""Voice-activity segmentation and transcription of mic ("me") and desktop audio."""

import io
import logging
import wave

import numpy as np

from ..events import Event
from ..model import SPEECH
from ..sources.audio import BLOCK, SAMPLE_RATE

log = logging.getLogger(__name__)
_BLOCK_S = BLOCK / SAMPLE_RATE
_PRE_ROLL = 3


def silero_vad():
    from pysilero_vad import SileroVoiceActivityDetector
    detector = SileroVoiceActivityDetector()

    def probability(block):
        pcm = (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16)
        return detector(pcm.tobytes())

    return probability


def to_wav(samples, rate=SAMPLE_RATE):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


class Segmenter:
    def __init__(self, vad, threshold=0.5, min_speech=0.3, silence=0.8, max_length=30.0):
        self.vad = vad
        self.threshold = threshold
        self.min_speech = min_speech
        self.silence = silence
        self.max_length = max_length
        self._pre_roll = []
        self._buffer = None
        self._speech_blocks = 0
        self._silent_blocks = 0

    def _finish(self):
        utterance, speech = self._buffer, self._speech_blocks
        self._buffer, self._speech_blocks, self._silent_blocks = None, 0, 0
        if speech * _BLOCK_S < self.min_speech:
            return []
        return [np.concatenate(utterance)]

    def feed(self, block):
        """Feed one BLOCK; returns any utterances that just finished."""
        speech = self.vad(block) >= self.threshold
        if self._buffer is None:
            if not speech:
                self._pre_roll = (self._pre_roll + [block])[-_PRE_ROLL:]
                return []
            self._buffer, self._pre_roll = self._pre_roll + [block], []
            self._speech_blocks, self._silent_blocks = 1, 0
            return []
        self._buffer.append(block)
        if speech:
            self._speech_blocks += 1
            self._silent_blocks = 0
        else:
            self._silent_blocks += 1
        if self._silent_blocks * _BLOCK_S >= self.silence:
            return self._finish()
        if len(self._buffer) * _BLOCK_S >= self.max_length:
            # Long monologue: cut here and keep listening without waiting for silence.
            done = self._finish()
            self._buffer, self._speech_blocks = [], 0
            return done
        return []


class SpeechProcessor:
    def __init__(self, client, publish, source):
        self.client = client
        self.publish = publish
        self.source = source
        self.kind = "speech.me" if source == "mic" else "speech.desktop"

    def handle(self, utterance, ts):
        future = self.client.transcribe(to_wav(utterance), SPEECH)
        future.add_done_callback(lambda done: self._done(done, ts))

    def _done(self, future, ts):
        try:
            text = future.result().strip()
        except Exception as exc:
            log.warning("Transcription failed for %s: %s", self.source, exc)
            return
        if text:
            self.publish(Event(ts, self.source, self.kind, text))
