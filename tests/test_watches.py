import asyncio
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.events import Event
from senses.watches import WatchRegistry, parse_condition


class WatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_watch_fires_on_matching_source_only(self):
        reg = WatchRegistry(clock=lambda: 0.0)
        w = reg.add(parse_condition({"text": "deployment (complete|failed)", "sources": ["screen"]}))
        self.assertEqual(reg.on_event(Event(1, "mic", "speech.me", "deployment complete")), [])
        self.assertEqual(reg.on_event(Event(2, "screen:desktop", "screen.text", "Deployment complete")), [w])
        self.assertEqual((await reg.wait(w.id, 0.1)).fired.ts, 2)
        self.assertEqual(reg.list(), [])

    async def test_wait_times_out(self):
        reg = WatchRegistry(clock=lambda: 0.0)
        w = reg.add(parse_condition({"presence": "present"}))
        self.assertIsNone(await reg.wait(w.id, 0.05))

    async def test_wait_wakes_when_fired_later(self):
        reg = WatchRegistry(clock=lambda: 0.0)
        w = reg.add(parse_condition({"presence": "present"}))
        waiter = asyncio.create_task(reg.wait(w.id, 1.0))
        await asyncio.sleep(0.01)
        reg.on_event(Event(5, "presence", "presence.present"))
        self.assertEqual((await waiter).fired.kind, "presence.present")

    async def test_unknown_watch(self):
        with self.assertRaises(KeyError):
            await WatchRegistry().wait("nope", 0.01)

    async def test_question_threshold_and_expiry(self):
        reg = WatchRegistry(clock=lambda: 0.0)
        w = reg.add(parse_condition({"question": "Is a progress bar visible?", "source": "screen"}), ttl=10)
        self.assertEqual(reg.question_watches(), [w])
        self.assertFalse(reg.report_score(w.id, 0.5, Event(1, "screen:desktop", "screen.text")))
        self.assertTrue(reg.has_screen_watch())
        self.assertEqual(reg.expire(11.0), [w])
        self.assertFalse(reg.has_screen_watch())

    async def test_question_fires_above_threshold(self):
        reg = WatchRegistry(clock=lambda: 0.0)
        w = reg.add(parse_condition({"question": "Is anyone there?", "source": "camera", "threshold": 0.6}))
        self.assertTrue(reg.report_score(w.id, 0.7, Event(1, "camera", "camera.scene")))
        self.assertEqual(reg.question_watches(), [])
        self.assertFalse(reg.has_screen_watch())

    def test_invalid_specs(self):
        with self.assertRaisesRegex(ValueError, "invalid regex"):
            parse_condition({"text": "("})
        with self.assertRaises(ValueError):
            parse_condition({"color": "red"})
        with self.assertRaises(ValueError):
            parse_condition({"presence": "sleeping"})

    def test_cap_ttl(self):
        reg = WatchRegistry(clock=lambda: 0.0)
        w = reg.add(parse_condition({"presence": "away"}), ttl=3600)
        reg.cap_ttl(1800, now=100.0)
        self.assertEqual(w.expires, 1900.0)


if __name__ == "__main__":
    unittest.main()
