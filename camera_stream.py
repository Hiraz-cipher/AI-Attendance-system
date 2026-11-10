"""One shared, reconnecting server-camera reader for all viewers."""
import os
import threading
import time
import cv2


def parse_source(value):
    value = str(value).strip()
    return int(value) if value.isdecimal() else value


class CameraStream:
    def __init__(self, source=2, backend="auto"):
        self.source = parse_source(source)
        self.backend = backend
        self.cap = None
        self._frame = None
        self._last_frame = 0
        self._lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.error = "Camera has not started."

    def start(self):
        with self._lifecycle:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()

    def _open(self, source):
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == 'nt' and isinstance(source, int) else [cv2.CAP_ANY]
        if os.name == "nt" and isinstance(source, int) and self.backend != "auto":
            backends = [cv2.CAP_MSMF if self.backend == "msmf" else cv2.CAP_DSHOW]
        for backend in backends:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def _update_loop(self):
        failures = 0
        try:
            while not self._stop.is_set():
                if self.cap is None:
                    self.error = 'Connecting to camera…'
                    self.cap = self._open(self.source)
                    if self.cap is None:
                        self.error = 'Camera unavailable. Check CAM_SOURCE, Windows camera permissions, and close other camera apps.'
                        self._stop.wait(2)
                        continue
                success, frame = self.cap.read()
                if success and frame is not None:
                    with self._lock:
                        self._frame, self._last_frame = frame, time.monotonic()
                    self.error = None
                    failures = 0
                    self._stop.wait(0.025)
                else:
                    failures += 1
                    if failures >= 10:
                        self.cap.release()
                        self.cap = None
                        with self._lock:
                            self._frame = None
                        self.error = 'Camera disconnected. Reconnecting…'
                        failures = 0
                        self._stop.wait(1)
                    else:
                        self._stop.wait(0.05)
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            with self._lock:
                self._frame = None

    def get_frame(self):
        with self._lock:
            if self._frame is None or time.monotonic() - self._last_frame > 3:
                return None
            return self._frame.copy()

    def is_open(self):
        with self._lock:
            return self._frame is not None and time.monotonic() - self._last_frame <= 3

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
