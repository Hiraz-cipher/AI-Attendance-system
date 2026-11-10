"""Hardware-free regression checks: python -m unittest discover -s tests."""
import sys
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

# Import the camera module without needing a native OpenCV installation.
cv_stub = SimpleNamespace(CAP_ANY=0, CAP_DSHOW=700, CAP_MSMF=1400, VideoCapture=None)
with patch.dict(sys.modules, cv2=cv_stub):
    import camera_stream

class Capture:
    def __init__(self, opened=True, frames=True):
        self.opened, self.frames, self.released = opened, frames, False
    def isOpened(self): return self.opened
    def read(self): return (True, np.ones((4,4,3))) if self.frames else (False, None)
    def release(self): self.released = True

class CameraTests(unittest.TestCase):
    def test_source_parsing(self):
        self.assertEqual(camera_stream.parse_source('2'), 2)
        self.assertEqual(camera_stream.parse_source('http://phone/video'), 'http://phone/video')
    def test_failed_handle_released(self):
        cap = Capture(opened=False)
        with patch.object(cv_stub, 'VideoCapture', return_value=cap):
            self.assertIsNone(camera_stream.CameraStream(0)._open(0))
        self.assertTrue(cap.released)
    def test_shared_reader_copy_and_cleanup(self):
        camera = camera_stream.CameraStream(0)
        cap = Capture()
        with patch.object(camera, '_open', return_value=cap) as opener:
            camera.start(); camera.start()
            deadline = time.monotonic()+1
            while not camera.is_open() and time.monotonic()<deadline: time.sleep(.01)
            frame = camera.get_frame()
            self.assertIsNotNone(frame)
            frame[:] = 0
            self.assertTrue(camera.get_frame().any())
            self.assertEqual(opener.call_count, 1)
            camera.stop()
        self.assertTrue(cap.released)
        self.assertIsNone(camera.get_frame())
    def test_stale_frame_rejected(self):
        camera = camera_stream.CameraStream(0)
        camera._frame = np.ones((2,2,3)); camera._last_frame = time.monotonic()-4
        self.assertIsNone(camera.get_frame())
        self.assertFalse(camera.is_open())
    def test_disconnect_clears_frame_and_reopens(self):
        camera = camera_stream.CameraStream(0)
        bad, good = Capture(frames=False), Capture()
        with patch.object(camera, '_open', side_effect=[bad, good]):
            camera.start()
            deadline = time.monotonic()+3
            while not camera.is_open() and time.monotonic()<deadline: time.sleep(.01)
            self.assertTrue(camera.is_open())
            self.assertTrue(bad.released)
            camera.stop()

if __name__ == '__main__': unittest.main()
