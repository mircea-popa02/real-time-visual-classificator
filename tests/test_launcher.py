import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.config import ModelConfig
from senses.launcher import detect_device, server_args, start_server, weights_cached


class LauncherTests(unittest.TestCase):
    def test_gpu_args_offline(self):
        args = server_args(ModelConfig(), "gpu", 8061, cached=True)
        for flag in ("--offline", "--reasoning-budget", "--image-max-tokens", "--jinja"):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("-ngl") + 1], "99")
        self.assertEqual(args[args.index("--port") + 1], "8061")
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertEqual(args[args.index("-np") + 1], "4")

    def test_cpu_args(self):
        args = server_args(ModelConfig(), "cpu", 8060, cached=False)
        self.assertNotIn("--offline", args)
        self.assertEqual(args[args.index("-ngl") + 1], "0")

    def test_detect_device(self):
        fake = lambda *a, **k: SimpleNamespace(stdout="Available devices:\n  Vulkan0: AMD Radeon (24560 MiB)\n", returncode=0)
        self.assertEqual(detect_device("llama-server", run=fake), "gpu")
        none = lambda *a, **k: SimpleNamespace(stdout="Available devices:\n", returncode=0)
        self.assertEqual(detect_device("llama-server", run=none), "cpu")

    def test_detect_device_missing_binary(self):
        def missing(*a, **k):
            raise FileNotFoundError("llama-server")
        self.assertEqual(detect_device("llama-server", run=missing), "cpu")

    def test_weights_cached(self):
        with tempfile.TemporaryDirectory() as hub:
            snap = Path(hub) / "models--ggml-org--gemma-4-E4B-it-GGUF" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            self.assertFalse(weights_cached("ggml-org/gemma-4-E4B-it-GGUF:Q4_0", env={"HF_HUB_CACHE": hub}))
            (snap / "gemma-4-E4B-it-Q4_0.gguf").write_bytes(b"")
            self.assertTrue(weights_cached("ggml-org/gemma-4-E4B-it-GGUF:Q4_0", env={"HF_HUB_CACHE": hub}))

    def test_configured_url_skips_launch(self):
        handle = start_server(ModelConfig(url="http://127.0.0.1:9999/v1"), Path("/tmp/unused.log"))
        self.assertEqual(handle.url, "http://127.0.0.1:9999/v1")
        handle.stop()


if __name__ == "__main__":
    unittest.main()
