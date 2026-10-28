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
    """Fan-out of events to asyncio queues; publish() is safe from any thread.

    The loop is the one subscribers live on, captured at subscribe(); with no subscribers publish() is a no-op.
    """

    def __init__(self):
        self._loop = None
        self._queues = []
        self._lock = threading.Lock()

    def subscribe(self, maxsize=1000):
        queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._loop = asyncio.get_running_loop()
            self._queues.append(queue)
        return queue

    def unsubscribe(self, queue):
        with self._lock:
            if queue in self._queues:
                self._queues.remove(queue)

    def publish(self, event):
        with self._lock:
            loop, subscribed = self._loop, bool(self._queues)
        if not subscribed or loop is None or loop.is_closed():
            return
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
