import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.events import Event
from senses.store import Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.s = Store(":memory:")
        self.s.add(Event(100.0, "screen:desktop", "screen.text", "Deployment complete in server.js"))
        self.s.add(Event(200.0, "mic", "speech.me", "trebuie să păstrăm datele"))
        self.s.add(Event(300.0, "screen:DP-1", "screen.text", "error: C++ build failed"))

    def tearDown(self):
        self.s.close()

    def test_punctuation_queries_do_not_raise(self):
        self.assertEqual([e.ts for e in self.s.search("server.js")], [100.0])
        self.assertEqual([e.ts for e in self.s.search('"C++" build')], [300.0])

    def test_diacritic_insensitive(self):
        self.assertEqual([e.ts for e in self.s.search("pastram")], [200.0])

    def test_source_prefix_and_time_filters(self):
        self.assertEqual([e.ts for e in self.s.search("error", sources=["screen"])], [300.0])
        self.assertEqual(self.s.search("error", until=250.0), [])
        self.assertEqual([e.ts for e in self.s.timeline(150.0)], [200.0, 300.0])

    def test_latest_and_purge(self):
        self.assertEqual(self.s.latest("screen.text").ts, 300.0)
        self.assertEqual(self.s.latest("screen.text", source="screen:desktop").ts, 100.0)
        self.assertEqual(self.s.purge(older_than=250.0), 2)
        self.assertEqual(self.s.search("pastram"), [])

    def test_empty_query(self):
        self.assertEqual(self.s.search("   "), [])

    def test_meta_round_trip_and_newest_first(self):
        self.s.add(Event(400.0, "screen:desktop", "screen.text", "build ok", {"busy": True}))
        results = self.s.search("build")
        self.assertEqual([e.ts for e in results], [400.0, 300.0])
        self.assertEqual(results[0].meta, {"busy": True})


if __name__ == "__main__":
    unittest.main()
