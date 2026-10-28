"""Change-driven screen OCR: only the band of the screen that changed is read again."""

from collections import Counter
from dataclasses import dataclass
import difflib
import time

import cv2
import numpy as np

from ..events import Event

_SCALE = 4
_BAND_PAD = 16
_LINE_MARGIN = 400


def changed_region(prev, cur, min_change=0.002, threshold=24):
    """(x, y, w, h) of what changed between two frames, or None when the change is negligible."""
    height, width = cur.shape[:2]
    if prev is None or prev.shape != cur.shape:
        return (0, 0, width, height)
    size = (max(1, width // _SCALE), max(1, height // _SCALE))
    small = [cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
             for image in (prev, cur)]
    mask = cv2.absdiff(small[0], small[1]) > threshold
    if mask.mean() < min_change or not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    x0, y0 = int(cols.min()) * _SCALE, int(rows.min()) * _SCALE
    x1 = min(width, (int(cols.max()) + 1) * _SCALE)
    y1 = min(height, (int(rows.max()) + 1) * _SCALE)
    return (x0, y0, x1 - x0, y1 - y0)


def _merge(boxes):
    boxes = [list(box) for box in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] <= b[0] + b[2] and b[0] <= a[0] + a[2] and a[1] <= b[1] + b[3] and b[1] <= a[1] + a[3]:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(box) for box in boxes]


def changed_boxes(prev, cur, min_change=0.002, threshold=24, margin=_LINE_MARGIN):
    """Separate boxes around each cluster of change, widened sideways so whole text lines are read.

    Changes on different monitors or far apart on one screen stay separate, keeping OCR input small.
    """
    height, width = cur.shape[:2]
    if prev is None or prev.shape != cur.shape:
        return [(0, 0, width, height)]
    size = (max(1, width // _SCALE), max(1, height // _SCALE))
    small = [cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
             for image in (prev, cur)]
    mask = (cv2.absdiff(small[0], small[1]) > threshold).astype(np.uint8)
    if not mask.any() or mask.mean() < min_change:
        return []
    # Join characters of one line into one blob before labelling.
    mask = cv2.dilate(mask, np.ones((3, 15), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for x, y, w, h, _ in stats[1:count]:
        x0 = max(0, x * _SCALE - margin)
        y0 = max(0, y * _SCALE - _BAND_PAD)
        x1 = min(width, (x + w) * _SCALE + margin)
        y1 = min(height, (y + h) * _SCALE + _BAND_PAD)
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return _merge(boxes)


@dataclass(frozen=True)
class OcrLine:
    text: str
    x: int
    y: int


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self):
        self._engine = None

    def read(self, image):
        if self._engine is None:
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
        result = self._engine(image)
        texts = list(result.txts or [])
        boxes = list(result.boxes) if result.boxes is not None else []
        return [OcrLine(text, int(box[0][0]), int(box[0][1])) for text, box in zip(texts, boxes) if text.strip()]


def new_lines(previous, current, similarity=1.0):
    """Lines of `current` that `previous` did not have, counting duplicates, in order.

    With similarity < 1, a line close enough to an unused previous line (OCR noise) is not new.
    """
    remaining = Counter(previous)
    fresh = []
    for line in current:
        if remaining[line]:
            remaining[line] -= 1
            continue
        if similarity < 1:
            match = next((old for old in remaining if remaining[old] and len(line) >= 8
                          and difflib.SequenceMatcher(None, old, line).ratio() >= similarity), None)
            if match is not None:
                remaining[match] -= 1
                continue
        fresh.append(line)
    return fresh


class _TokenBucket:
    def __init__(self, per_minute, clock):
        self.capacity = max(1, per_minute)
        self.rate = per_minute / 60.0
        self.tokens = float(self.capacity)
        self.clock = clock
        self.last = clock()

    def take(self):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class ScreenTextProcessor:
    def __init__(self, ocr, min_change, ocr_per_minute, clock=time.monotonic):
        self.ocr = ocr
        self.min_change = min_change
        self.ocr_per_minute = ocr_per_minute
        self.clock = clock
        self._images = {}
        self._lines = {}
        self._buckets = {}
        self._busy = {}

    def lines(self, screen):
        return [line.text for line in self._lines.get(screen, [])]

    def process(self, frame):
        screen = frame.name
        boxes = changed_boxes(self._images.get(screen), frame.image, self.min_change)
        if not boxes:
            return None
        bucket = self._buckets.setdefault(screen, _TokenBucket(self.ocr_per_minute, self.clock))
        if not bucket.take():
            # Keep the old reference so the skipped change is still seen next time.
            self._busy[screen] = True
            return None
        self._images[screen] = frame.image
        if screen not in self._lines:
            height, width = frame.image.shape[:2]
            boxes = [(0, 0, width, height)]
        lines = self._lines.get(screen, [])
        fresh = []
        for x, y, w, h in boxes:
            inside = lambda line: x <= line.x < x + w and y <= line.y < y + h
            read = [OcrLine(line.text, line.x + x, line.y + y) for line in self.ocr.read(frame.image[y:y + h, x:x + w])]
            fresh += new_lines([line.text for line in lines if inside(line)], [line.text for line in read],
                               similarity=0.85)
            lines = [line for line in lines if not inside(line)] + read
        self._lines[screen] = sorted(lines, key=lambda line: (line.y, line.x))
        busy = self._busy.pop(screen, False)
        if not fresh:
            return None
        return Event(frame.ts, f"screen:{screen}", "screen.text", "\n".join(fresh), {"screen": screen, "busy": busy})
