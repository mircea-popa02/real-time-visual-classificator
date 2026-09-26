import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Also support: python tests/test_decision.py from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_vision.catalog import object_questions
from jev_vision.decision import (DecisionError, answer_for, jev_decide, local_decide,
                                 local_discover, probabilities_for)
from jev_vision.cli import build_parser, _decide, print_result


class DecisionTests(unittest.TestCase):
    def test_probability_and_multiple_objects(self):
        candidates = [{"token": "A", "logprob": math.log(3)},
                      {"token": "B", "logprob": 0.0}]
        self.assertAlmostEqual(probabilities_for(candidates, {"true": None, "false": None})["true"], .75)
        self.assertEqual(answer_for({"type": "noul"}, {"true": .75, "false": .25})["noul"], .75)
        self.assertEqual(len(object_questions(["dog", "cat"])), 2)

    def test_missing_candidate_fails(self):
        with self.assertRaises(DecisionError):
            probabilities_for([{"token": "A", "logprob": 0}], {"true": None, "false": None})

    @patch("jev_vision.decision._post")
    def test_local_request_image_and_scores(self, post):
        post.return_value = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "A", "logprob": -0.2}, {"token": "B", "logprob": -1.7}]}]}}]}
        result = local_decide("Inspect", object_questions(["dog"]), ["data:image/jpeg;base64,AAA"],
                              "http://localhost:8080/v1", "vision")
        self.assertGreater(result["answers"]["dog"]["noul"], .8)
        url, body = post.call_args.args[:2]
        self.assertEqual(url, "http://localhost:8080/v1/chat/completions")
        self.assertEqual(body["messages"][0]["content"][1]["type"], "image_url")

    @patch.dict("os.environ", {"JEV_API_KEY": "test-key"})
    @patch("jev_vision.decision._post")
    def test_jev_native_request(self, post):
        post.return_value = {"code": 0, "data": {"answers": {"dog": {"type": "noul", "noul": .2}}}}
        result = jev_decide({"inventory": ["dog"]}, object_questions(["dog"]))
        self.assertEqual(result["answers"]["dog"]["noul"], .2)
        self.assertEqual(post.call_args.args[0], "https://www.jevai.org/api/v1/decisions")
        self.assertEqual(post.call_args.args[2], "test-key")

    @patch("jev_vision.decision._post")
    def test_discovery_shortlist_is_validated(self, post):
        post.return_value = {"choices": [{"message": {"content":
            '{"objects":["chair","cat","chair","spaceship"]}'}}]}
        self.assertEqual(local_discover(["data:image/jpeg;base64,AAA"], ["chair", "cat"],
                                        "http://localhost:8080/v1", "vision"), ["chair", "cat"])
        body = post.call_args.args[1]
        self.assertEqual(body["response_format"]["schema"]["properties"]["objects"]["items"]["enum"],
                         ["chair", "cat"])

    @patch("jev_vision.cli.local_decide")
    @patch("jev_vision.cli.local_discover", return_value=["chair", "cat"])
    def test_default_webcam_discovers_then_scores_only_shortlist(self, discover, decide):
        args = build_parser().parse_args([])
        _decide(args, "Inspect", object_questions(["person"]), ["data:image/jpeg;base64,AAA"])
        self.assertEqual(list(decide.call_args.args[1]), ["chair", "cat"])
        self.assertEqual(decide.call_args.args[4], "local-vision")


if __name__ == "__main__":
    unittest.main()
