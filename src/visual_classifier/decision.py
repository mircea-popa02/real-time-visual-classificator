"""Structured vision observations and local token-probability scoring."""

import json
import math
import concurrent.futures
import urllib.error
import urllib.request


# Reasoning models (e.g. Gemma 4) must answer directly: scoring reads the first output token.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


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
    if kind == "binary":
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
    if kind == "binary":
        return {"type": "binary", "probability": probabilities["true"]}
    if kind == "choice":
        return {"type": kind, "choice": max(probabilities, key=probabilities.get), "probabilities": probabilities}
    return {"type": kind, "score": sum(int(key) * value for key, value in probabilities.items()),
            "legend": {str(i): label for i, label in enumerate(question["criteria"])},
            "probabilities": probabilities}


def local_decide(state, questions, images, base_url, model, api_key=None, timeout=90,
                 on_answer=None, parallel=2):
    validate_questions(questions)
    if not model:
        raise DecisionError("Set --model to the model alias served by llama.cpp")
    def ask_one(name, question):
        # Images first: every question shares the same prompt prefix, so the server reuses its KV cache.
        content = [{"type": "image_url", "image_url": {"url": image}} for image in images]
        content += [{"type": "text", "text": prompt_for(state, question)}]
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "max_completion_tokens": 1, "temperature": 0, "logprobs": True,
                "top_logprobs": 128, **NO_THINKING}
        response = _post(base_url.rstrip("/") + "/chat/completions", body, api_key, timeout)
        try:
            candidates = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DecisionError(f"Model returned no token logprobs for question {name!r}") from exc
        probabilities = probabilities_for(candidates, options_for(question))
        return name, answer_for(question, probabilities)

    completed = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(parallel, len(questions))) as pool:
        futures = [pool.submit(ask_one, name, question) for name, question in questions.items()]
        for future in concurrent.futures.as_completed(futures):
            name, answer = future.result()
            completed[name] = answer
            if on_answer:
                on_answer(name, answer)
    return {"answers": {name: completed[name] for name in questions}}


SETTINGS = ("indoors", "outdoors", "unclear")
LIGHTING = ("dark", "dim", "bright", "unclear")


def observation_grammar(catalog=None, limit=6, describe=False):
    """GBNF for one compact line: objects | setting | light [| sentence].

    A terse line instead of JSON keeps generation to roughly 15 tokens, which dominates frame latency.
    """
    if catalog:
        name = " | ".join(json.dumps(item) for item in catalog)
    else:
        name = '[a-z] [a-z -]{0,23}'
    alternatives = lambda values: " | ".join(json.dumps(value) for value in values)
    return "\n".join([
        'root ::= objs " | " setting " | " light' + (' " | " sentence' if describe else ""),
        f'objs ::= "none" | name (", " name){{0,{limit - 1}}}',
        f"name ::= {name}",
        f"setting ::= {alternatives(SETTINGS)}",
        f"light ::= {alternatives(LIGHTING)}",
        "sentence ::= [A-Z] [a-zA-Z0-9 ,'-]{10,200} \".\"",
    ])


def local_observe(images, base_url, model, catalog=None, api_key=None, timeout=90, limit=6, describe=False):
    """Name the visible objects, setting, and light (optionally a sentence) in one short model request.

    Without a catalog any object name is allowed; with one, names are constrained to it.
    """
    if not images:
        raise DecisionError("Object discovery needs an image")
    if not model:
        raise DecisionError("Set --model to the model alias served by llama.cpp")
    names = list(dict.fromkeys(catalog)) if catalog is not None else None
    if names is not None and not names:
        raise DecisionError("Object catalog is empty")
    vocabulary = (f"ONLY from this catalog: {', '.join(names)}" if names
                  else "(physical things, not places or qualities), as short lowercase nouns")
    text = (f"Reply on one line: up to {limit} distinct objects that are clearly visible {vocabulary}, "
            "most prominent first, or none | indoors, outdoors or unclear | dark, dim, bright or unclear"
            + (" | a short factual sentence describing the scene" if describe else "")
            + ". Do not infer identities or hidden objects.")
    content = [{"type": "image_url", "image_url": {"url": image}} for image in images]
    content += [{"type": "text", "text": text}]
    body = {
        "model": model, "messages": [{"role": "user", "content": content}],
        "temperature": 0, "max_completion_tokens": 32 + 8 * limit + (64 if describe else 0),
        "grammar": observation_grammar(names, limit, describe), **NO_THINKING,
    }
    response = _post(base_url.rstrip("/") + "/chat/completions", body, api_key, timeout)
    try:
        parts = [part.strip() for part in response["choices"][0]["message"]["content"].strip().split(" | ")]
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise DecisionError("Model did not return an observation") from exc
    if len(parts) < 3:
        raise DecisionError(f"Model returned an unreadable observation: {' | '.join(parts)!r}")
    # Validate even when the server ignores the grammar.
    selected = []
    for item in parts[0].split(","):
        item = item.strip(" -").lower()
        if item and item != "none" and item not in selected and (names is None or item in names):
            selected.append(item)
    return {
        "objects": selected[:limit],
        "setting": parts[1] if parts[1] in SETTINGS else "unclear",
        "lighting": parts[2] if parts[2] in LIGHTING else "unclear",
        "summary": " | ".join(parts[3:])[:240] if describe else "",
    }
