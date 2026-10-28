import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.config import Config, Rule
from senses.events import Event
from senses.rules import RulesEngine, resolve_action

AWAY = Event(0.0, "presence", "presence.away")
BACK = Event(0.0, "presence", "presence.present")
which = lambda name: f"/usr/bin/{name}"


class RulesTests(unittest.TestCase):
    def setUp(self):
        self.ran = []
        self.engine = RulesEngine(Config().rules, run=self.ran.append, which=which)

    def test_away_fires_after_ten_seconds(self):
        self.engine.on_event(AWAY, now=0.0)
        self.engine.tick(9.9)
        self.assertEqual(self.ran, [])
        events = self.engine.tick(10.0)
        self.assertEqual(self.ran, [["playerctl", "-a", "pause"], ["loginctl", "lock-session"]])
        self.assertEqual([e.kind for e in events], ["rule.ran", "rule.ran"])
        self.engine.tick(30.0)
        self.assertEqual(len(self.ran), 2)

    def test_return_cancels_pending(self):
        self.engine.on_event(AWAY, now=0.0)
        self.engine.on_event(BACK, now=5.0)
        self.engine.tick(20.0)
        self.assertEqual(self.ran, [])

    def test_hold_defers_lock_only(self):
        self.engine.hold("@screen.lock")
        self.engine.on_event(AWAY, now=0.0)
        self.engine.tick(10.0)
        self.assertEqual(self.ran, [["playerctl", "-a", "pause"]])
        self.engine.release("@screen.lock", now=50.0)
        self.assertEqual(self.ran[-1], ["loginctl", "lock-session"])

    def test_release_after_return_does_nothing(self):
        self.engine.hold("@screen.lock")
        self.engine.on_event(AWAY, now=0.0)
        self.engine.tick(10.0)
        self.engine.on_event(BACK, now=20.0)
        self.engine.release("@screen.lock", now=30.0)
        self.assertNotIn(["loginctl", "lock-session"], self.ran)

    def test_suspended_ignores_presence(self):
        self.engine.on_event(AWAY, now=0.0)
        self.engine.suspended = True
        self.engine.tick(20.0)
        self.engine.on_event(AWAY, now=21.0)
        self.engine.tick(40.0)
        self.assertEqual(self.ran, [])

    def test_pause_all_event_cancels_pending(self):
        self.engine.on_event(AWAY, now=0.0)
        self.engine.on_event(Event(1.0, "system", "capture.paused", meta={"all": True}), now=1.0)
        self.engine.tick(20.0)
        self.assertEqual(self.ran, [])

    def test_match_and_placeholders(self):
        engine = RulesEngine([Rule(on="screen.text", match=r"build failed", run=["notify-send Build {text}"])],
                             run=self.ran.append, which=which)
        engine.on_event(Event(0.0, "screen:desktop", "screen.text", "all good"), now=0.0)
        engine.on_event(Event(0.0, "screen:desktop", "screen.text", "build failed"), now=0.0)
        engine.tick(0.0)
        self.assertEqual(self.ran, [["notify-send", "Build", "build failed"]])

    def test_fallback_and_missing(self):
        only_xdg = lambda n: "/usr/bin/xdg-screensaver" if n == "xdg-screensaver" else None
        self.assertEqual(resolve_action("@screen.lock", AWAY, which=only_xdg), ["xdg-screensaver", "lock"])
        self.assertIsNone(resolve_action("@media.pause", AWAY, which=lambda n: None))


if __name__ == "__main__":
    unittest.main()
