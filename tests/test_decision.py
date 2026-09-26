import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Also support: python tests/test_decision.py from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_classifier.catalog import object_questions
from visual_classifier.decision import (DecisionError, answer_for, local_decide,
                                        local_observe, probabilities_for)
from visual_classifier.cli import build_parser, _decide


class DecisionTests(unittest.TestCase):
    def test_probability_and_multiple_objects(self):
        candidates = [{"token": "A", "logprob": math.log(3)},
                      {"token": "B", "logprob": 0.0}]
        self.assertAlmostEqual(probabilities_for(candidates, {"true": None, "false": None})["true"], .75)
        self.assertEqual(answer_for({"type": "binary"}, {"true": .75, "false": .25})["probability"], .75)
        self.assertEqual(len(object_questions(["dog", "cat"])), 2)

    def test_missing_candidate_fails(self):
        with self.assertRaises(DecisionError):
            probabilities_for([{"token": "A", "logprob": 0}], {"true": None, "false": None})

    @patch("visual_classifier.decision._post")
    def test_local_request_image_and_scores(self, post):
        post.return_value = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "A", "logprob": -0.2}, {"token": "B", "logprob": -1.7}]}]}}]}
        result = local_decide("Inspect", object_questions(["dog"]), ["data:image/jpeg;base64,AAA"],
                              "http://localhost:8080/v1", "vision")
        self.assertGreater(result["answers"]["dog"]["probability"], .8)
        url, body = post.call_args.args[:2]
        self.assertEqual(url, "http://localhost:8080/v1/chat/completions")
        self.assertEqual(body["messages"][0]["content"][1]["type"], "image_url")

    @patch("visual_classifier.decision._post")
    def test_structured_observation_is_validated(self, post):
        post.return_value = {"choices": [{"message": {"content":
            '{"objects":["chair","cat","chair","spaceship"],"setting":"indoors",'
            '"lighting":"bright","summary":"A chair and a cat."}'}}]}
        observation = local_observe(["data:image/jpeg;base64,AAA"], ["chair", "cat"],
                                    "http://localhost:8080/v1", "vision")
        self.assertEqual(observation["objects"], ["chair", "cat"])
        self.assertEqual(observation["setting"], "indoors")
        body = post.call_args.args[1]
        self.assertEqual(body["response_format"]["schema"]["properties"]["objects"]["items"]["enum"],
                         ["chair", "cat"])

    @patch("visual_classifier.cli.local_decide")
    @patch("visual_classifier.cli.local_observe", return_value={"objects": ["chair", "cat"],
                                                       "setting": "indoors", "lighting": "bright", "summary": "Room"})
    def test_default_webcam_discovers_then_scores_only_shortlist(self, discover, decide):
        args = build_parser().parse_args([])
        _decide(args, "Inspect", object_questions(["person"]), ["data:image/jpeg;base64,AAA"])
        self.assertEqual(list(decide.call_args.args[1]), ["chair", "cat"])
        self.assertEqual(decide.call_args.args[4], "local-vision")
        self.assertFalse(args.preview)
        self.assertEqual(discover.call_args.kwargs["limit"], 4)

    @patch("visual_classifier.cli.local_decide")
    @patch("visual_classifier.cli.local_observe", return_value={"objects": ["person"],
                                                       "setting": "indoors", "lighting": "bright", "summary": "Person"})
    def test_progress_reaches_terminal_before_frame_finishes(self, discover, decide):
        events = []
        decide.side_effect = lambda *args, **kwargs: (
            kwargs["on_answer"]("person", {"type": "binary", "probability": .8})
            or {"answers": {"person": {"type": "binary", "probability": .8}}})
        result = _decide(build_parser().parse_args([]), "Inspect", object_questions(["person"]),
                         ["data:image/jpeg;base64,AAA"], lambda kind, payload: events.append(kind))
        self.assertEqual(events, ["observation", "answer"])
        self.assertEqual(result["answers"]["person"]["probability"], .8)


if __name__ == "__main__":
    unittest.main()
