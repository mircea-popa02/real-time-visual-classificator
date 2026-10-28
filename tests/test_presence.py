import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.config import PresenceConfig
from senses.presence import PresenceFSM, PresenceState, idle_probability


class PresenceTests(unittest.TestCase):
    def setUp(self):
        self.fsm = PresenceFSM(PresenceConfig())

    def test_enter_needs_sustained_probability(self):
        self.assertIsNone(self.fsm.update(0.9, 0.0))
        self.assertIsNone(self.fsm.update(0.9, 1.0))
        self.assertEqual(self.fsm.update(0.9, 2.0), PresenceState.PRESENT)

    def test_leave_after_five_seconds_and_flap_resets(self):
        self.fsm.update(0.9, 0)
        self.fsm.update(0.9, 2)
        self.fsm.update(0.1, 10)
        self.fsm.update(0.5, 12)
        self.assertIsNone(self.fsm.update(0.1, 14))
        self.assertIsNone(self.fsm.update(0.1, 18))
        self.assertEqual(self.fsm.update(0.1, 19), PresenceState.AWAY)

    def test_model_error_holds_state(self):
        self.fsm.update(0.9, 0)
        self.fsm.update(0.9, 2)
        for t in range(3, 60):
            self.assertIsNone(self.fsm.update(None, float(t)))
        self.assertEqual(self.fsm.state, PresenceState.PRESENT)

    def test_error_resets_pending_leave(self):
        self.fsm.update(0.9, 0)
        self.fsm.update(0.9, 2)
        self.fsm.update(0.1, 10)
        self.fsm.update(None, 12)
        self.assertIsNone(self.fsm.update(0.1, 15))
        self.assertEqual(self.fsm.update(0.1, 20), PresenceState.AWAY)

    def test_interval_fast_then_slow(self):
        self.fsm.update(0.9, 0)
        self.fsm.update(0.9, 2)
        self.assertEqual(self.fsm.interval(10), 1.0)
        self.assertAlmostEqual(self.fsm.interval(100), 1 / 0.3)
        self.fsm.update(0.1, 100)
        self.assertEqual(self.fsm.interval(101), 1.0)

    def test_idle_probability(self):
        self.assertEqual(idle_probability(5.0), 1.0)
        self.assertEqual(idle_probability(120.0), 0.0)
        self.assertIsNone(idle_probability(None))


if __name__ == "__main__":
    unittest.main()
