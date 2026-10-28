import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.config import ConfigError, load_config, parse_duration


def write_tmp(text):
    handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    handle.write(text)
    handle.close()
    return Path(handle.name)


class ConfigTests(unittest.TestCase):
    def test_defaults_without_file(self):
        cfg = load_config(Path("/nonexistent/config.toml"))
        self.assertEqual(cfg.screen.interval, 2.0)
        self.assertEqual(cfg.store.retention_days, 30)
        self.assertEqual(cfg.rules[0].on, "presence.away")
        self.assertEqual(cfg.rules[0].after, 10.0)
        self.assertEqual(cfg.rules[0].run, ["@media.pause", "@screen.lock"])
        self.assertEqual(cfg.server.host, "127.0.0.1")
        self.assertEqual(cfg.audio.languages, ["Romanian", "English"])

    def test_toml_overrides_and_durations(self):
        tmp = write_tmp('[screen]\ninterval = "500ms"\n[[rules]]\non = "presence.present"\nrun = ["@notify"]\n')
        cfg = load_config(tmp)
        self.assertEqual(cfg.screen.interval, 0.5)
        self.assertEqual([r.on for r in cfg.rules], ["presence.present"])

    def test_unknown_key_is_reported(self):
        with self.assertRaisesRegex(ConfigError, "screen.intervall"):
            load_config(write_tmp("[screen]\nintervall = 2\n"))

    def test_unknown_section_is_reported(self):
        with self.assertRaisesRegex(ConfigError, "screens"):
            load_config(write_tmp("[screens]\ninterval = 2\n"))

    def test_parse_duration(self):
        self.assertEqual(parse_duration("2m"), 120.0)
        self.assertEqual(parse_duration(3), 3.0)
        with self.assertRaises(ConfigError):
            parse_duration("soon")


if __name__ == "__main__":
    unittest.main()
