from contextlib import redirect_stderr, redirect_stdout
import io
import sys
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.cli import _datasette_argv, main
from senses.config import Config, StoreConfig


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


class ViewTests(unittest.TestCase):
    def test_uvx_is_preferred(self):
        with patch("senses.cli.shutil.which", side_effect=lambda name: "/usr/bin/uvx" if name == "uvx" else None):
            self.assertEqual(_datasette_argv(Path("/tmp/senses.db"), "127.0.0.1", 8001),
                             ["/usr/bin/uvx", "datasette", "/tmp/senses.db", "--host", "127.0.0.1", "--port", "8001"])

    def test_standalone_datasette(self):
        with patch("senses.cli.shutil.which",
                   side_effect=lambda name: "/usr/bin/datasette" if name == "datasette" else None):
            self.assertEqual(_datasette_argv(Path("/tmp/senses.db"), "127.0.0.1", 8001),
                             ["/usr/bin/datasette", "/tmp/senses.db", "--host", "127.0.0.1", "--port", "8001"])

    def test_no_launcher(self):
        with patch("senses.cli.shutil.which", return_value=None):
            self.assertIsNone(_datasette_argv(Path("/tmp/senses.db"), "127.0.0.1", 8001))

    def test_missing_database(self):
        cfg = Config(store=StoreConfig(path="/nonexistent/senses.db"))
        with patch("senses.cli.load_config", return_value=cfg):
            with redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main(["view"]), 1)
        self.assertIn("does not exist", err.getvalue())

    def test_execs_datasette(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "senses.db"
            path.touch()
            cfg = Config(store=StoreConfig(path=str(path)))
            with patch("senses.cli.load_config", return_value=cfg), \
                    patch("senses.cli.shutil.which",
                          side_effect=lambda name: "/usr/bin/uvx" if name == "uvx" else None), \
                    patch("senses.cli.os.execvp") as execvp:
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["view", "--port", "9000"]), 0)
        self.assertEqual(execvp.call_args.args[1],
                         ["/usr/bin/uvx", "datasette", str(path), "--host", "127.0.0.1", "--port", "9000"])


if __name__ == "__main__":
    unittest.main()
