import sys
from pathlib import Path
import asyncio
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.events import Event, EventBus


class EventBusTests(unittest.IsolatedAsyncioTestCase):
    async def test_publish_from_thread_reaches_subscriber(self):
        bus = EventBus()
        q = bus.subscribe()
        await asyncio.to_thread(bus.publish, Event(1.0, "mic", "speech.me", "salut"))
        self.assertEqual((await asyncio.wait_for(q.get(), 1)).text, "salut")

    async def test_full_queue_drops_oldest(self):
        bus = EventBus()
        q = bus.subscribe(maxsize=2)
        for i in range(3):
            bus.publish(Event(float(i), "mic", "speech.me", str(i)))
        await asyncio.sleep(0)
        self.assertEqual([q.get_nowait().text, q.get_nowait().text], ["1", "2"])

    async def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        q = bus.subscribe()
        bus.unsubscribe(q)
        bus.publish(Event(1.0, "mic", "speech.me", "x"))
        await asyncio.sleep(0)
        self.assertTrue(q.empty())


if __name__ == "__main__":
    unittest.main()
