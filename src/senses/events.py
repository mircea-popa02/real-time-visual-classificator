"""Events and the in-process event bus."""

import asyncio
from dataclasses import dataclass, field
import threading


@dataclass(frozen=True)
class Event:
    """One observation. Sources: screen:<name>, camera, mic, desktop, presence, system."""

    ts: float
    source: str
    kind: str
    text: str = ""
    meta: dict = field(default_factory=dict)


class EventBus:
    """Fan-out of events to asyncio queues; publish() is safe from any thread."""

    def __init__(self, loop=None):
        self._loop = loop
        self._queues = []
        self._lock = threading.Lock()

    def _get_loop(self):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return self._loop

    def subscribe(self, maxsize=1000):
        self._get_loop()
        queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._queues.append(queue)
        return queue

    def unsubscribe(self, queue):
        with self._lock:
            if queue in self._queues:
                self._queues.remove(queue)

    def publish(self, event):
        loop = self._get_loop() if self._loop is None else self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._deliver(event)
        else:
            loop.call_soon_threadsafe(self._deliver, event)

    def _deliver(self, event):
        with self._lock:
            queues = list(self._queues)
        for queue in queues:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(event)
