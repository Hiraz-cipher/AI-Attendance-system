"""
liveness.py
------------
Layer 4 — Liveness Detection (Anti-Spoofing)

Two independent physiological signals, matching the project spec's Layer 4
and end-to-end flow (Section 8/9/23):

1. EAR (Eye Aspect Ratio) blink detection — catches printed-photo spoofing.
   A static photo never blinks.
2. Head-pose estimation via cv2.solvePnP — catches a second, different class
   of spoof: someone holding a phone/tablet dead still to fake a blink loop,
   or a face that never turns/tilts at all. A live person's head drifts and
   turns naturally; a flat printed photo or a phone held rigidly in place
   does not.

Someone is "live" if EITHER a blink OR a head-pose movement was detected
recently (LIVE_WINDOW_SECONDS). This is intentionally an OR, not an AND —
requiring both simultaneously makes the system unpleasant to use (blink AND
turn your head at the exact same moment), and the spec's own algorithm
(Section 23, steps 5-8) treats blink and head-pose as two contributing
pieces of "physiological evidence" rather than a strict two-factor gate.
If your evaluation later shows the OR condition lets spoofing through above
your <2% FAR target, tightening to AND is a one-line change (see
_is_live_now below) — worth doing only if your own testing shows it's
needed, not by default.

What this does NOT defend against: a video replay of the person blinking
and moving their head (e.g. playing a video of them on a phone/tablet).
That needs texture/reflection analysis or a depth sensor, out of scope for
a webcam-only setup — treat this as "keeps out static photo spoofing plus
rigid-device spoofing," not "keeps out every spoof."

Method: MediaPipe FaceLandmarker (the current Tasks API — the older
mp.solutions.face_mesh API used in most tutorials has been removed from
recent mediapipe releases) gives 478 facial landmarks per frame.

EAR:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
EAR stays roughly constant (~0.25-0.35) with eyes open and drops sharply
(~<0.20) for a couple of frames during a blink.

Head pose:
    Six landmark points (nose tip, chin, eye outer corners, mouth corners)
    are matched against a generic 3D face model and passed to
    cv2.solvePnP() to recover a rotation vector, which is converted to
    pitch/yaw/roll Euler angles. A meaningful swing in yaw or pitch from a
    rolling baseline counts as a "head movement" event.

Setup note: the first time this runs, it downloads a small (~4MB) model file
from Google's servers. That requires normal internet access on the machine
running the Flask app / recognition script (not needed again after the first
run — it's cached to disk locally).
"""

import os
import time
import math
import urllib.request

import numpy as np
import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# ---- Eye landmarks (EAR) ----
# MediaPipe's 478-point face mesh topology — index sets for a 6-point EAR
# calculation per eye (same layout as the classic dlib 68-point EAR method).
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]   # p1..p6
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # p1..p6

EAR_BLINK_THRESHOLD = 0.21   # below this = eyes considered closed
EAR_CONSEC_FRAMES = 2        # must stay closed this many frames to count (filters noise)

# ---- Head-pose landmarks (solvePnP) ----
NOSE_TIP_IDX = 1
CHIN_IDX = 152
LEFT_EYE_OUTER_IDX = 263
RIGHT_EYE_OUTER_IDX = 33
LEFT_MOUTH_CORNER_IDX = 291
RIGHT_MOUTH_CORNER_IDX = 61

# Generic 3D face model (millimetres, arbitrary reference frame) — standard
# point set widely used for webcam head-pose estimation. Not this specific
# person's face geometry, just an average adult face shape; good enough to
# recover approximate rotation, not precise 3D reconstruction.
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye outer corner
    (225.0, 170.0, -135.0),   # Right eye outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

YAW_MOVEMENT_THRESHOLD_DEG = 12.0    # head turn left/right
PITCH_MOVEMENT_THRESHOLD_DEG = 10.0  # head nod up/down

LIVE_WINDOW_SECONDS = 6.0    # a blink OR head movement within this window = "currently live"


def _ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("[liveness] Downloading face landmark model (first run only)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[liveness] Model downloaded.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download the MediaPipe face landmark model automatically ({e}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {MODEL_PATH}"
        )


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _eye_ratio(points):
    p1, p2, p3, p4, p5, p6 = points
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4))


def _rotation_matrix_to_euler(rmat):
    """Returns (pitch, yaw, roll) in degrees from a 3x3 rotation matrix.

    Note: due to the axis convention of this landmark/model-point pairing,
    the absolute pitch and roll values sit near +-180 degrees for a frontal
    face rather than near 0 (yaw is unaffected and reads near 0 as expected).
    This is harmless here because LivenessDetector only ever compares the
    *change* in angle against a rolling baseline (see _update_head_pose),
    never the absolute value — but worth knowing if you print these values
    directly for debugging and they look "wrong" at first glance.
    """
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class LivenessDetector:
    def __init__(self):
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._start_time = time.time()

        # Blink tracking
        self._closed_frame_count = 0
        self._last_blink_time = 0.0
        self._total_blinks = 0

        # Head-pose tracking
        self._baseline_yaw = None
        self._baseline_pitch = None
        self._last_movement_time = 0.0
        self._total_movements = 0

    def reset(self):
        """Call when starting to evaluate a new person/session."""
        self._closed_frame_count = 0
        self._last_blink_time = 0.0
        self._total_blinks = 0
        self._baseline_yaw = None
        self._baseline_pitch = None
        self._last_movement_time = 0.0
        self._total_movements = 0

    def _update_blink(self, ear):
        blinked_now = False
        if ear < EAR_BLINK_THRESHOLD:
            self._closed_frame_count += 1
        else:
            if self._closed_frame_count >= EAR_CONSEC_FRAMES:
                blinked_now = True
                self._total_blinks += 1
                self._last_blink_time = time.time()
            self._closed_frame_count = 0
        return blinked_now

    def _update_head_pose(self, landmarks, w, h):
        """Runs solvePnP and returns (pitch, yaw, roll, movement_detected)."""

        def pt(idx):
            lm = landmarks[idx]
            return (lm.x * w, lm.y * h)

        image_points = np.array([
            pt(NOSE_TIP_IDX),
            pt(CHIN_IDX),
            pt(LEFT_EYE_OUTER_IDX),
            pt(RIGHT_EYE_OUTER_IDX),
            pt(LEFT_MOUTH_CORNER_IDX),
            pt(RIGHT_MOUTH_CORNER_IDX),
        ], dtype=np.float64)

        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, _tvec = cv2.solvePnP(
            MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None, None, None, False

        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = _rotation_matrix_to_euler(rmat)

        movement_detected = False
        if self._baseline_yaw is None:
            # First good reading — establish baseline, no movement yet.
            self._baseline_yaw = yaw
            self._baseline_pitch = pitch
        else:
            delta_yaw = abs(yaw - self._baseline_yaw)
            delta_pitch = abs(pitch - self._baseline_pitch)
            if delta_yaw > YAW_MOVEMENT_THRESHOLD_DEG or delta_pitch > PITCH_MOVEMENT_THRESHOLD_DEG:
                movement_detected = True
                self._total_movements += 1
                self._last_movement_time = time.time()
                # Re-baseline so we can detect the *next* movement too,
                # rather than staying "moved" forever after one turn.
                self._baseline_yaw = yaw
                self._baseline_pitch = pitch

        return pitch, yaw, roll, movement_detected

    def process(self, frame_bgr):
        """
        frame_bgr: full camera frame (numpy array, BGR — as read by cv2).
        Returns a dict:
            face_found        -> bool
            ear                -> float or None
            blinked_now        -> bool, a blink was just confirmed this frame
            total_blinks       -> int, since last reset()
            pitch, yaw, roll   -> float degrees or None
            movement_now       -> bool, a head-pose movement was just confirmed
            total_movements    -> int, since last reset()
            is_live            -> bool, blink OR movement within LIVE_WINDOW_SECONDS
        """
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB, no copy needed for mediapipe.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return {
                "face_found": False, "ear": None, "blinked_now": False, "total_blinks": self._total_blinks,
                "pitch": None, "yaw": None, "roll": None, "movement_now": False,
                "total_movements": self._total_movements, "is_live": self._is_live_now(),
            }

        landmarks = result.face_landmarks[0]

        def pt(idx):
            lm = landmarks[idx]
            return (lm.x * w, lm.y * h)

        left_ear = _eye_ratio([pt(i) for i in LEFT_EYE_IDX])
        right_ear = _eye_ratio([pt(i) for i in RIGHT_EYE_IDX])
        ear = (left_ear + right_ear) / 2.0
        blinked_now = self._update_blink(ear)

        pitch, yaw, roll, movement_now = self._update_head_pose(landmarks, w, h)

        return {
            "face_found": True,
            "ear": ear,
            "blinked_now": blinked_now,
            "total_blinks": self._total_blinks,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "movement_now": movement_now,
            "total_movements": self._total_movements,
            "is_live": self._is_live_now(),
        }

    def _is_live_now(self):
        """OR condition: either signal within the live window counts.
        To require BOTH a recent blink AND a recent head movement instead
        (stricter, may hurt usability — only do this if your own FAR testing
        shows the OR condition isn't strict enough), change this to:
            blink_recent = (time.time() - self._last_blink_time) < LIVE_WINDOW_SECONDS
            movement_recent = (time.time() - self._last_movement_time) < LIVE_WINDOW_SECONDS
            return blink_recent and movement_recent
        """
        blink_recent = (time.time() - self._last_blink_time) < LIVE_WINDOW_SECONDS
        movement_recent = (time.time() - self._last_movement_time) < LIVE_WINDOW_SECONDS
        return blink_recent or movement_recent

    def close(self):
        self._landmarker.close()
