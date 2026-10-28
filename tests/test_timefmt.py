import sys
from pathlib import Path
import unittest
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.events import Event
from senses.timefmt import format_events, parse_since


class TimefmtTests(unittest.TestCase):
    def test_parse_since_relative_and_iso(self):
        self.assertEqual(parse_since("10m", now=1000.0), 400.0)
        self.assertAlmostEqual(parse_since("2026-10-28T18:00:00", now=0), datetime(2026, 10, 28, 18).timestamp())
        with self.assertRaisesRegex(ValueError, "10m"):
            parse_since("yesterday-ish", now=0)

    def test_format_events_multiline_and_empty(self):
        ts = datetime(2026, 10, 28, 18, 16, 2).timestamp()
        out = format_events([Event(ts, "screen:desktop", "screen.text", "a\nb")], now=ts)
        self.assertEqual(out, "18:16:02  screen:desktop  a\n" + " " * 26 + "b")
        self.assertEqual(format_events([]), "(nothing)")

    def test_presence_and_older_dates(self):
        ts = datetime(2026, 10, 27, 9, 0, 0).timestamp()
        now = datetime(2026, 10, 28, 9, 0, 0).timestamp()
        self.assertEqual(format_events([Event(ts, "presence", "presence.away")], now=now),
                         "2026-10-27 09:00:00  presence        away")


if __name__ == "__main__":
    unittest.main()
