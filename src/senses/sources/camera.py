"""Webcam frames through OpenCV; the device comes from config or the first index that opens."""

import time

import cv2

from .screen import Frame, SourceError


class Camera:
    name = "opencv"

    def __init__(self, cfg, opener=cv2.VideoCapture):
        self.cfg = cfg
        self._opener = opener
        self._capture = None

    def _open(self):
        devices = [self.cfg.device] if self.cfg.device is not None else range(4)
        for device in devices:
            capture = self._opener(device)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return capture
            capture.release()
        raise SourceError("No camera could be opened")

    def read(self):
        if self._capture is None:
            self._capture = self._open()
        ok, image = self._capture.read()
        if not ok or image is None:
            self.close()
            raise SourceError("Could not read a camera frame")
        return Frame("camera", image, time.time())

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
