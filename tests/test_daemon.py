import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import FakeCamera, FakeClient, FakeOcr, FakeScreen
from senses.config import Config
from senses.daemon import CaptureState, Daemon
from senses.store import Store
from senses.watches import parse_condition


class DaemonTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.t = [1000.0]
        self.ran = []
        self.screen, self.camera, self.client = FakeScreen(), FakeCamera(), FakeClient()
        self.ocr = FakeOcr({0: "hello", 50: "secret", 100: "Deployment complete", 150: "later"})
        self.store = Store(":memory:")
        self.d = Daemon(Config(), self.store, self.client, self.screen, self.camera, (None, None), self.ocr,
                        clock=lambda: self.t[0], run=self.ran.append, which=lambda n: f"/usr/bin/{n}")

    async def advance(self, seconds, step=1.0):
        end = self.t[0] + seconds
        while self.t[0] < end:
            self.t[0] += step
            await self.d.presence_step()
            self.d.housekeeping()

    async def become_present(self):
        self.client.presence = 0.9
        await self.advance(3)
        self.assertEqual(self.d.status()["presence"], "present")

    def kinds(self):
        return [e.kind for e in self.store.timeline(0)]

    async def test_present_screen_text_is_stored(self):
        await self.become_present()
        await self.d.screen_step()
        self.assertEqual([e.text for e in self.store.search("hello")], ["hello"])
        self.assertIn("presence.present", self.kinds())

    async def test_pause_stops_capture_but_presence_runs(self):
        await self.become_present()
        await self.d.screen_step()
        self.d.pause()
        self.screen.value = 50
        await self.d.screen_step()
        self.assertEqual(self.store.search("secret"), [])
        self.client.presence = 0.1
        await self.advance(20)
        self.assertEqual(self.d.status()["presence"], "away")
        self.assertIn(["loginctl", "lock-session"], self.ran)
        self.assertEqual(self.d.status()["capture"], CaptureState.PAUSED.value)

    async def test_pause_all_suspends_rules(self):
        await self.become_present()
        self.d.pause(all=True)
        reads = self.camera.reads
        self.client.presence = 0.1
        await self.advance(30)
        self.assertEqual(self.ran, [])
        self.assertEqual(self.camera.reads, reads)
        self.d.resume()
        self.assertEqual(self.d.status()["capture"], "active")

    async def test_timed_pause_resumes(self):
        await self.become_present()
        self.d.pause(duration=5)
        await self.advance(6)
        self.assertEqual(self.d.status()["capture"], "active")
        self.assertIn("capture.resumed", self.kinds())

    async def test_away_screen_watch_not_stored_and_holds_lock(self):
        await self.become_present()
        await self.d.screen_step()
        watch = self.d.watches.add(parse_condition({"text": "deployment complete", "sources": ["screen"]}))
        self.client.presence = 0.1
        await self.advance(20)
        self.assertEqual(self.d.status()["presence"], "away")
        self.assertEqual(self.ran, [["playerctl", "-a", "pause"]])
        self.assertTrue(self.d.status()["lock_held_by_watch"])
        self.screen.value = 100
        await self.d.screen_step()
        self.assertEqual(self.store.search("Deployment"), [e for e in self.store.search("Deployment")
                                                           if e.kind == "watch.fired"])
        self.assertTrue(any(e.kind == "watch.fired" for e in self.store.search("Deployment")))
        self.d.housekeeping()
        self.assertEqual(self.ran[-1], ["loginctl", "lock-session"])
        self.assertEqual((await self.d.watches.wait(watch.id, 0.1)).fired.text, "Deployment complete")

    async def test_away_without_watch_does_not_capture_screen(self):
        await self.become_present()
        self.client.presence = 0.1
        await self.advance(20)
        captures = self.screen.captures
        await self.d.screen_step()
        self.assertEqual(self.screen.captures, captures)

    async def test_failing_source_turns_off_and_recovers(self):
        await self.become_present()
        self.screen.failures = 3
        for _ in range(3):
            await self.d.screen_step()
        status = self.d.status()
        self.assertEqual(status["sources"]["screen"]["state"], "off")
        self.assertIn("screen gone", status["sources"]["screen"]["error"])
        self.assertIn("source.off", self.kinds())
        captures = self.screen.captures
        await self.d.screen_step()
        self.assertEqual(self.screen.captures, captures)
        self.t[0] += 61
        await self.d.screen_step()
        self.assertEqual(self.d.status()["sources"]["screen"]["state"], "on")
        self.assertIn("source.on", self.kinds())

    async def test_model_down_never_locks(self):
        await self.become_present()
        self.client.down = True
        await self.advance(120)
        self.assertEqual(self.d.status()["presence"], "present")
        self.assertEqual(self.ran, [])

    async def test_question_watch_on_screen(self):
        await self.become_present()
        watch = self.d.watches.add(parse_condition({"question": "Is a progress bar visible?", "source": "screen"}))
        self.client.scores[f"watch:{watch.id}"] = 0.95
        await self.d.screen_step()
        self.assertIsNotNone((await self.d.watches.wait(watch.id, 0.1)))

    async def test_now_and_status_shape(self):
        await self.become_present()
        await self.d.screen_step()
        now = self.d.now()
        self.assertEqual(now["screens"]["desktop"]["text"], "hello")
        self.assertEqual(now["presence"], "present")
        status = self.d.status()
        self.assertEqual(set(status), {"presence", "capture", "paused_until", "sources", "model", "watches",
                                       "lock_held_by_watch", "exclusions"})

    async def test_ask_and_look_use_fresh_capture(self):
        result = await self.d.ask("camera", {"cup": {"type": "binary", "instructions": "Is there a cup?"}})
        self.assertIn("cup", result["answers"])
        self.assertIn("A man sits.", await self.d.look("camera"))
        self.assertEqual(await self.d.look("screen", prompt="What app?"), "Konsole: testing")


if __name__ == "__main__":
    unittest.main()
