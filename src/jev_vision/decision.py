"""Request validation, local logprob scoring, and the native Jev adapter."""

import json
import math
import os
import urllib.error
import urllib.request


class DecisionError(Exception):
    """A malformed request, transport error, or unusable model response."""


def _post(url, body, token=None, timeout=90):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, ValueError) as exc:
        raise DecisionError(f"Request to {url} failed: {exc}") from exc


def options_for(question):
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        options = {"true": None, "false": None}
    elif kind == "choice" and isinstance(criteria, dict):
        options = criteria
    elif kind == "score" and isinstance(criteria, list):
        options = {str(i): label for i, label in enumerate(criteria)}
    else:
        raise DecisionError(f"Invalid question type or criteria: {kind!r}")
    if not 2 <= len(options) <= 20:
        raise DecisionError("Each question needs 2 to 20 options")
    if not isinstance(question.get("instructions"), str):
        raise DecisionError("Question instructions must be text")
    return options


def validate_questions(questions):
    if not isinstance(questions, dict) or not questions:
        raise DecisionError("questions must be a nonempty object")
    for question in questions.values():
        if not isinstance(question, dict):
            raise DecisionError("Each question must be an object")
        options_for(question)


def prompt_for(state, question):
    options = options_for(question)
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    lines = [f"State:\n{state_text}\n\nQuestion: {question['instructions']}\nOptions:"]
    for letter, (key, description) in zip("ABCDEFGHIJKLMNOPQRST", options.items()):
        lines.append(f"[{letter}] {key}" + (f": {description}" if description is not None else ""))
    return "\n".join(lines) + "\n\nAnswer with the letter of the best option only."


def probabilities_for(candidates, keys):
    letters = "ABCDEFGHIJKLMNOPQRST"[:len(keys)]
    found = {}
    for candidate in candidates:
        token = candidate.get("token")
        value = candidate.get("logprob")
        if token in letters and isinstance(value, (float, int)) and math.isfinite(value) and value > -9999:
            found[token] = max(value, found.get(token, float("-inf")))
    missing = [letter for letter in letters if letter not in found]
    if not found:
        raise DecisionError("No option tokens found in top logprobs; try a different model")
    peak = max(found.values())
    weights = [math.exp(found[letter] - peak) if letter in found else 0 for letter in letters]
    total = sum(weights)
    if missing:
        # Omitted alternatives can have up to the lowest returned candidate's weight.
        finite = [c["logprob"] for c in candidates if isinstance(c.get("logprob"), (float, int)) and math.isfinite(c["logprob"]) and c["logprob"] > -9999]
        if not finite or len(missing) * math.exp(min(finite) - peak) / total >= 1e-6:
            raise DecisionError(f"Model omitted potentially significant option scores: {', '.join(missing)}")
    return {key: weight / total for key, weight in zip(keys, weights)}


def answer_for(question, probabilities):
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probabilities["true"]}
    if kind == "choice":
        return {"type": kind, "choice": max(probabilities, key=probabilities.get), "probabilities": probabilities}
    return {"type": kind, "score": sum(int(key) * value for key, value in probabilities.items()),
            "legend": {str(i): label for i, label in enumerate(question["criteria"])},
            "probabilities": probabilities}


def local_decide(state, questions, images, base_url, model, api_key=None, timeout=90):
    validate_questions(questions)
    if not model:
        raise DecisionError("Set --model to the model alias served by llama.cpp")
    answers = {}
    for name, question in questions.items():
        content = [{"type": "text", "text": prompt_for(state, question)}]
        content += [{"type": "image_url", "image_url": {"url": image}} for image in images]
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "max_completion_tokens": 1, "temperature": 0, "logprobs": True,
                "top_logprobs": 128}
        response = _post(base_url.rstrip("/") + "/chat/completions", body, api_key, timeout)
        try:
            candidates = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DecisionError(f"Model returned no token logprobs for question {name!r}") from exc
        probabilities = probabilities_for(candidates, options_for(question))
        answers[name] = answer_for(question, probabilities)
    return {"answers": answers}


def local_discover(images, catalog, base_url, model, api_key=None, timeout=90, limit=8):
    """Ask the vision model for a shortlist from the catalog before scoring it."""
    if not images:
        raise DecisionError("Object discovery needs an image")
    if not model:
        raise DecisionError("Set --model to the model alias served by llama.cpp")
    names = list(dict.fromkeys(catalog))
    if not names:
        raise DecisionError("Object catalog is empty")
    content = [{"type": "text", "text":
                "Identify clearly visible objects in this image. Select up to "
                f"{limit} distinct names ONLY from this catalog: {', '.join(names)}. "
                "Return an empty list if none is visible. Do not infer hidden objects."}]
    content += [{"type": "image_url", "image_url": {"url": image}} for image in images]
    body = {
        "model": model, "messages": [{"role": "user", "content": content}],
        "temperature": 0, "max_completion_tokens": 160,
        "response_format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {"objects": {"type": "array", "items": {
                "type": "string", "enum": names}, "maxItems": limit}},
            "required": ["objects"], "additionalProperties": False}},
    }
    response = _post(base_url.rstrip("/") + "/chat/completions", body, api_key, timeout)
    try:
        data = json.loads(response["choices"][0]["message"]["content"])
        proposals = data["objects"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DecisionError("Model did not return a valid object shortlist") from exc
    if not isinstance(proposals, list):
        raise DecisionError("Model returned an invalid object shortlist")
    # Validate even when the server does not enforce response_format.
    allowed = set(names)
    selected = []
    for item in proposals:
        if isinstance(item, str) and item in allowed and item not in selected:
            selected.append(item)
    return selected[:limit]


def jev_decide(state, questions, model=None, base_url="https://www.jevai.org", timeout=90):
    validate_questions(questions)
    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise DecisionError("Set JEV_API_KEY in the environment")
    body = {"state": state, "questions": questions}
    if model:
        body["model"] = model
    response = _post(base_url.rstrip("/") + "/api/v1/decisions", body, key, timeout)
    if response.get("code") != 0 or not isinstance(response.get("data", {}).get("answers"), dict):
        raise DecisionError(f"Jev rejected the request: {response.get('message', response)}")
    return response["data"]
