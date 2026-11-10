"""
face_engine.py
----------------
Shared face-detection + recognition logic used by both app.py (web dashboard)
and recognize_and_log.py (standalone desktop script).

Why this exists:
- Your original app.py and recognize_and_log.py had near-identical DeepFace
  logic copy-pasted in both files. Any fix (bug, model swap, tuning) had to be
  made twice. This module is the single source of truth.

Efficiency changes vs. the original:
1. No temp .jpg written to disk every 2 seconds. DeepFace.find() accepts an
   in-memory numpy array directly, so we skip the imwrite/imread round trip
   entirely (this was your biggest hidden cost — disk I/O + JPEG encode/decode
   on every recognition cycle).
2. detector_backend="skip" tells DeepFace "trust the crop I'm giving you,
   don't re-run your own face detector on it." You already detected + cropped
   the face with Haar Cascade, so asking DeepFace to detect again was wasted
   work happening twice per face, every cycle.
3. DeepFace.build_model() is called once at startup instead of relying on
   DeepFace to lazily build it on the first frame, which caused a visible
   freeze on the first face seen.
"""

import cv2
import os
from deepface import DeepFace


class FaceEngine:
    def __init__(self, dataset_dir="dataset", model_name="VGG-Face", cascade_path=None):
        self.dataset_dir = dataset_dir
        self.model_name = model_name

        cascade_path = cascade_path or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade at {cascade_path}")

        # Warm up the recognition model once at startup instead of on first use.
        DeepFace.build_model(model_name)

    def detect_faces(self, frame, scale_factor=1.1, min_neighbors=5, min_size=(100, 100)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )

    def identify(self, face_roi):
        """
        face_roi: a cropped BGR numpy array (already cropped by Haar Cascade).
        Returns the matched person's name, or None if no match / no dataset entry.
        """
        if face_roi is None or face_roi.size == 0:
            return None

        try:
            dfs = DeepFace.find(
                img_path=face_roi,
                db_path=self.dataset_dir,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
            if len(dfs) > 0 and not dfs[0].empty:
                matched_path = dfs[0].iloc[0]["identity"]
                return os.path.basename(os.path.dirname(matched_path))
        except Exception:
            pass
        return None

    def rebuild_embeddings(self):
        """Regenerates the cached embeddings (.pkl) file DeepFace keeps inside
        the dataset folder, by running find() against the first image found.
        Call this after adding/removing student photos so newly registered
        faces are actually recognized — DeepFace.find() otherwise keeps using
        a stale cache. Shared by encode_faces.py and the web registration
        endpoint so both stay in sync with the exact same logic."""
        first_img = None
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    first_img = os.path.join(root, file)
                    break
            if first_img:
                break

        if not first_img:
            return False

        try:
            DeepFace.find(
                img_path=first_img,
                db_path=self.dataset_dir,
                model_name=self.model_name,
                enforce_detection=False,
            )
        except Exception:
            pass  # DeepFace can raise after the cache is already rebuilt successfully
        return True
