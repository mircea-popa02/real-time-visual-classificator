"""Short model-written notes: what the user is doing on a screen, and what the camera sees."""

import base64
import concurrent.futures

import cv2

from ..events import Event
from ..model import BACKGROUND

ACTIVITY_PROMPT = ("In under 15 words, in the form '<application>: <what the user is doing>', "
                   "for example 'Firefox: reading a GitHub README'.")


def activity_grammar():
    """`<application>: <what the user is doing>`, short and without markdown."""
    return "root ::= [A-Z] [^\\n|*:]{1,30} \": \" [a-z] [^\\n|*]{4,100}"


def frame_uri(frame, max_edge=1600):
    image = frame.image
    height, width = image.shape[:2]
    if max(height, width) > max_edge:
        scale = max_edge / max(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Could not encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()


class Throttle:
    def __init__(self, min_interval):
        self.min_interval = min_interval
        self._last = None

    def ready(self, now):
        if self._last is not None and now - self._last < self.min_interval:
            return False
        self._last = now
        return True


def _then(future, fn):
    result = concurrent.futures.Future()

    def done(source):
        try:
            result.set_result(fn(source.result()))
        except BaseException as exc:
            result.set_exception(exc)

    future.add_done_callback(done)
    return result


def summarize(client, frame):
    future = client.describe([frame_uri(frame)], ACTIVITY_PROMPT, BACKGROUND,
                             grammar=activity_grammar(), max_tokens=48)
    return _then(future, lambda text: Event(frame.ts, f"screen:{frame.name}", "screen.summary", text.strip()))


def scene_note(client, frame):
    future = client.observe([frame_uri(frame)], BACKGROUND, describe=True)

    def event(observation):
        details = f"{', '.join(observation['objects']) or 'nothing named'}; " \
                  f"{observation['setting']}, {observation['lighting']}"
        text = f"{observation['summary']} ({details})" if observation.get("summary") else f"({details})"
        return Event(frame.ts, "camera", "camera.scene", text)

    return _then(future, event)
