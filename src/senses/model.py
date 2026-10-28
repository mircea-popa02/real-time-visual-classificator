"""One priority-queued client for every model call, so interactive and presence requests never wait behind
summaries or transcription backlog."""

import concurrent.futures
import itertools
import logging
import queue
import threading
import urllib.request

from . import decision
from .decision import DecisionError

INTERACTIVE = 0
PRESENCE = 1
WATCH = 2
SPEECH = 3
BACKGROUND = 4

log = logging.getLogger(__name__)


class ModelClient:
    def __init__(self, base_url, model, workers=4, timeout=30.0):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.grammar_supported = True
        self._jobs = queue.PriorityQueue()
        self._sequence = itertools.count()
        self._threads = [threading.Thread(target=self._work, daemon=True, name=f"model-{i}")
                         for i in range(workers)]
        for thread in self._threads:
            thread.start()

    def submit(self, priority, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        self._jobs.put((priority, next(self._sequence), future, fn, args, kwargs))
        return future

    def _call(self, fn, args, kwargs):
        kwargs = {"base_url": self.base_url, "model": self.model, "timeout": self.timeout, **kwargs}
        if not self.grammar_supported:
            kwargs.pop("grammar", None)
        try:
            return fn(*args, **kwargs)
        except DecisionError as exc:
            # Servers without llama.cpp grammars reject the field; retry once and stop sending it.
            if "grammar" in kwargs and "400" in str(exc):
                log.warning("Server rejected the grammar field; continuing without grammars")
                self.grammar_supported = False
                kwargs.pop("grammar")
                return fn(*args, **kwargs)
            raise

    def _work(self):
        while True:
            _, _, future, fn, args, kwargs = self._jobs.get()
            if future is None:
                return
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(self._call(fn, args, kwargs))
            except BaseException as exc:
                future.set_exception(exc)

    def ask(self, images, questions, state, priority):
        return self.submit(priority, lambda base_url, model, timeout: decision.local_decide(
            state, questions, images, base_url, model, timeout=timeout))

    def observe(self, images, priority, **kwargs):
        return self.submit(priority, decision.local_observe, images, **kwargs)

    def describe(self, images, prompt, priority, **kwargs):
        return self.submit(priority, decision.describe, images, prompt, **kwargs)

    def transcribe(self, wav, priority):
        return self.submit(priority, decision.transcribe, wav)

    def healthy(self):
        root = self.base_url.rstrip("/").removesuffix("/v1")
        try:
            with urllib.request.urlopen(root + "/health", timeout=1) as response:
                return response.status == 200
        except OSError:
            return False

    def close(self):
        for _ in self._threads:
            self._jobs.put((float("inf"), next(self._sequence), None, None, (), {}))
