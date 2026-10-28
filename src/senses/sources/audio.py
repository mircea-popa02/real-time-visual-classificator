"""Microphone and desktop-audio capture through `soundcard` (PulseAudio/PipeWire monitors, WASAPI loopback)."""

import logging

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 512  # samples; 32 ms, the chunk size Silero VAD expects at 16 kHz

log = logging.getLogger(__name__)


def _soundcard():
    # Imported lazily: soundcard connects to the audio server at import time.
    import soundcard
    return soundcard


def _named(sc, name, loopback):
    return sc.get_microphone(name, include_loopback=loopback)


def find_devices(cfg, sc=None):
    """(mic, desktop) devices from config names or the system defaults; either may be None."""
    sc = sc or _soundcard()
    try:
        mic = _named(sc, cfg.mic, False) if cfg.mic else sc.default_microphone()
    except Exception as exc:
        log.warning("No microphone: %s", exc)
        mic = None
    try:
        speaker = cfg.desktop or sc.default_speaker().name
        desktop = _named(sc, speaker, True)
    except Exception as exc:
        log.warning("No desktop audio loopback: %s", exc)
        desktop = None
    return mic, desktop


class AudioStream:
    def __init__(self, device, name):
        self.device = device
        self.name = name
        self._closed = False

    def blocks(self):
        """Yield mono float32 blocks of BLOCK samples until close()."""
        with self.device.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK) as recorder:
            while not self._closed:
                data = recorder.record(numframes=BLOCK)
                yield (data.mean(axis=1) if data.ndim == 2 else data).astype(np.float32)

    def close(self):
        self._closed = True
