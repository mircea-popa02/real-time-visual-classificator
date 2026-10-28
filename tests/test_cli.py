from contextlib import redirect_stderr, redirect_stdout
import io
import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.cli import main


class CliTests(unittest.TestCase):
    def test_not_running_message(self):
        with patch("senses.cli._control", side_effect=ConnectionRefusedError):
            with redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main(["status"]), 1)
        self.assertIn("senses is not running", err.getvalue())

    def test_pause_sends_duration_and_level(self):
        with patch("senses.cli._control", return_value={"capture": "paused_all", "paused_until": None}) as control:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["pause", "10m", "--all"]), 0)
        self.assertEqual(control.call_args.args[:3], ("POST", "/control/pause", {"duration": "10m", "all": True}))

    def test_search_prints_formatted_events(self):
        events = [{"ts": time.time(), "source": "mic", "kind": "speech.me", "text": "salut lume"}]
        with patch("senses.cli._control", return_value=events) as control:
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["search", "salut", "--since", "1h"]), 0)
        self.assertIn("mic             salut lume", out.getvalue())
        self.assertEqual(control.call_args.args[:2], ("GET", "/control/search?q=salut&since=1h&limit=20"))

    def test_resume(self):
        with patch("senses.cli._control", return_value={"capture": "active", "paused_until": None}) as control:
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["resume"]), 0)
        self.assertEqual(control.call_args.args[:2], ("POST", "/control/resume"))
        self.assertIn("active", out.getvalue())


if __name__ == "__main__":
    unittest.main()
