import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Also support: python tests/test_decision.py from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senses.decision import (DecisionError, describe, transcribe, answer_for, local_decide,
                                        local_observe, probabilities_for)


class DecisionTests(unittest.TestCase):
    def test_probability_and_multiple_objects(self):
        candidates = [{"token": "A", "logprob": math.log(3)},
                      {"token": "B", "logprob": 0.0}]
        self.assertAlmostEqual(probabilities_for(candidates, {"true": None, "false": None})["true"], .75)
        self.assertEqual(answer_for({"type": "binary"}, {"true": .75, "false": .25})["probability"], .75)

    def test_missing_candidate_fails(self):
        with self.assertRaises(DecisionError):
            probabilities_for([{"token": "A", "logprob": 0}], {"true": None, "false": None})

    @patch("senses.decision._post")
    def test_local_request_image_and_scores(self, post):
        post.return_value = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "A", "logprob": -0.2}, {"token": "B", "logprob": -1.7}]}]}}]}
        result = local_decide("Inspect", {"dog": {"type": "binary", "instructions": "Is a dog visible?"}}, ["data:image/jpeg;base64,AAA"],
                              "http://localhost:8080/v1", "vision")
        self.assertGreater(result["answers"]["dog"]["probability"], .8)
        url, body = post.call_args.args[:2]
        self.assertEqual(url, "http://localhost:8080/v1/chat/completions")
        self.assertEqual(body["messages"][0]["content"][0]["type"], "image_url")
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])

    @patch("senses.decision._post")
    def test_catalog_observation_is_validated(self, post):
        post.return_value = {"choices": [{"message": {"content": "chair, cat, chair, spaceship | indoors | glowing"}}]}
        observation = local_observe(["data:image/jpeg;base64,AAA"], "http://localhost:8080/v1", "vision",
                                    catalog=["chair", "cat"])
        self.assertEqual(observation, {"objects": ["chair", "cat"], "setting": "indoors",
                                       "lighting": "unclear", "summary": ""})
        body = post.call_args.args[1]
        self.assertIn('name ::= "chair" | "cat"', body["grammar"])
        self.assertEqual(body["messages"][0]["content"][0]["type"], "image_url")
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])

    @patch("senses.decision._post")
    def test_open_observation_with_description(self, post):
        post.return_value = {"choices": [{"message": {"content":
            "man, coffee mug, Man | indoors | dim | A man holds a coffee mug at a desk."}}]}
        observation = local_observe(["data:image/jpeg;base64,AAA"], "http://localhost:8080/v1", "vision",
                                    describe=True, limit=3)
        self.assertEqual(observation["objects"], ["man", "coffee mug"])
        self.assertEqual(observation["summary"], "A man holds a coffee mug at a desk.")
        grammar = post.call_args.args[1]["grammar"]
        self.assertIn('objs ::= "none" | name (", " name){0,2}', grammar)
        self.assertIn('" | " sentence', grammar)

    @patch("senses.decision._post")
    def test_observation_none_and_garbage(self, post):
        post.return_value = {"choices": [{"message": {"content": "none | outdoors | dark"}}]}
        self.assertEqual(local_observe(["data:image/jpeg;base64,AAA"], "http://x/v1", "vision")["objects"], [])
        post.return_value = {"choices": [{"message": {"content": "I cannot see"}}]}
        with self.assertRaises(DecisionError):
            local_observe(["data:image/jpeg;base64,AAA"], "http://x/v1", "vision")


class TranscribeTests(unittest.TestCase):
    @patch("senses.decision._post")
    def test_transcribe_request(self, post):
        post.return_value = {"choices": [{"message": {"content": "  salut lume \n"}}]}
        self.assertEqual(transcribe(b"RIFF....", "http://x/v1", "m"), "salut lume")
        body = post.call_args.args[1]
        self.assertEqual(body["messages"][0]["content"][0]["type"], "input_audio")
        self.assertEqual(body["messages"][0]["content"][0]["input_audio"]["format"], "wav")
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])

    @patch("senses.decision._post")
    def test_transcribe_restricts_languages(self, post):
        post.return_value = {"choices": [{"message": {"content": "buna"}}]}
        transcribe(b"RIFF", "http://x/v1", "m", languages=["Romanian", "English"])
        prompt = post.call_args.args[1]["messages"][0]["content"][1]["text"]
        self.assertIn("Romanian", prompt)
        self.assertIn("English", prompt)

    @patch("senses.decision._post")
    def test_transcribe_malformed_response(self, post):
        post.return_value = {"choices": []}
        with self.assertRaises(DecisionError):
            transcribe(b"RIFF", "http://x/v1", "m")

    @patch("senses.decision._post")
    def test_describe_passes_grammar_and_image_first(self, post):
        post.return_value = {"choices": [{"message": {"content": "Konsole | running tests"}}]}
        self.assertEqual(describe(["data:image/jpeg;base64,AAA"], "What?", "http://x/v1", "m", grammar="root ::= x"),
                         "Konsole | running tests")
        body = post.call_args.args[1]
        self.assertEqual(body["grammar"], "root ::= x")
        self.assertEqual(body["messages"][0]["content"][0]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
