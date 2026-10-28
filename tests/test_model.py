import sys
from pathlib import Path
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.decision import DecisionError
from senses.model import BACKGROUND, PRESENCE, ModelClient


class ModelClientTests(unittest.TestCase):
    def setUp(self):
        self.client = ModelClient("http://x/v1", "m", workers=1)

    def tearDown(self):
        self.client.close()

    def test_higher_priority_runs_first(self):
        gate, order = threading.Event(), []
        self.client.submit(BACKGROUND, lambda **kw: gate.wait(1))
        a = self.client.submit(BACKGROUND, lambda **kw: order.append("summary"))
        b = self.client.submit(PRESENCE, lambda **kw: order.append("presence"))
        gate.set()
        a.result(1)
        b.result(1)
        self.assertEqual(order, ["presence", "summary"])

    def test_submit_passes_connection_kwargs(self):
        seen = self.client.submit(PRESENCE, lambda **kw: kw).result(1)
        self.assertEqual((seen["base_url"], seen["model"]), ("http://x/v1", "m"))

    def test_grammar_dropped_after_http_400(self):
        calls = []

        def fn(**kw):
            calls.append("grammar" in kw)
            if "grammar" in kw:
                raise DecisionError("Request failed: HTTP Error 400: Bad Request")
            return "ok"

        self.assertEqual(self.client.submit(PRESENCE, fn, grammar="root ::= x").result(1), "ok")
        self.assertFalse(self.client.grammar_supported)
        self.client.submit(PRESENCE, fn, grammar="root ::= x").result(1)
        self.assertEqual(calls, [True, False, False])

    def test_errors_propagate_to_future(self):
        def fail(**kw):
            raise DecisionError("down")
        with self.assertRaises(DecisionError):
            self.client.submit(PRESENCE, fail).result(1)


if __name__ == "__main__":
    unittest.main()
