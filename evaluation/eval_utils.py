"""
eval_utils.py
--------------
Shared configuration and helper functions for the evaluation/ scripts.

This folder is deliberately self-contained: it imports from the parent
project's liveness.py when needed, but never touches the parent project's
dataset/, attendance.db, or app.py. Nothing here modifies the running
application — it only reads from evaluation/eval_data/ and writes results
into evaluation/results/.

Folder layout (eval_data/):

    eval_data/
    ├── recognition/
    │   ├── gallery/          <person>/*.jpg   -- 80% split, "registered" set
    │   └── probe/             <person>/*.jpg   -- 20% split, held-out test set
    │                          (nearest-neighbour comparison, no training —
    │                           see evaluate_recognition.py)
    │
    ├── classifier/
    │   ├── train/              <person>/*.jpg  -- trained-classifier training set
    │   ├── validation/         <person>/*.jpg  -- optional, held out from training
    │   └── test/                <person>/*.jpg  -- trained-classifier test set
    │                          (SVM / KNN — see evaluate_recognition_svm.py /
    │                           evaluate_recognition_knn.py)
    │
    ├── liveness/
    │   ├── genuine/             <person>/(clip.mp4 or frames)  -- real/live samples
    │   └── spoof/
    │       ├── printed_photo/   <person>/(clip.mp4 or frames)
    │       └── screen_replay/   <person>/(clip.mp4 or frames)
    │
    ├── lighting/
    │   ├── bright/               <person>/*.jpg
    │   ├── normal/                <person>/*.jpg
    │   └── low_light/           <person>/*.jpg
    │
    └── pose/
        ├── frontal/               <person>/*.jpg
        ├── left/                   <person>/*.jpg
        ├── right/                  <person>/*.jpg
        ├── up/                     <person>/*.jpg
        └── down/                   <person>/*.jpg

IMPORTANT: recognition/ and classifier/ are deliberately separate datasets.
recognition/gallery+probe feeds the untrained, nearest-neighbour distance
comparison (evaluate_recognition.py). classifier/train+test feeds the
*trained* SVM/KNN classifiers (evaluate_recognition_svm.py /
evaluate_recognition_knn.py). Reusing the same images for both would let
the classifier "see" gallery images that are supposed to be reserved for
the untrained comparison, so keep them as distinct capture sessions.
"""

import os
import sys
import time
import csv

# Allow importing liveness.py / face_engine.py from the parent project folder
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

EVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVAL_ROOT, "eval_data")
RESULTS_DIR = os.path.join(EVAL_ROOT, "results")

# --- 1. Recognition: Gallery + Probe (untrained nearest-neighbour comparison) ---
GALLERY_DIR = os.path.join(DATA_DIR, "recognition", "gallery")   # 80% split — "registered" images per person
PROBE_DIR = os.path.join(DATA_DIR, "recognition", "probe")       # 20% split — held-out test images per person

# --- 2. Classifier: Train / Validation / Test (trained SVM & KNN) ---
CLASSIFIER_TRAIN_DIR = os.path.join(DATA_DIR, "classifier", "train")
CLASSIFIER_VAL_DIR = os.path.join(DATA_DIR, "classifier", "validation")
CLASSIFIER_TEST_DIR = os.path.join(DATA_DIR, "classifier", "test")

# --- 3. Liveness: Genuine + Spoof ---
GENUINE_DIR = os.path.join(DATA_DIR, "liveness", "genuine")
SPOOF_PHOTO_DIR = os.path.join(DATA_DIR, "liveness", "spoof", "printed_photo")
SPOOF_SCREEN_DIR = os.path.join(DATA_DIR, "liveness", "spoof", "screen_replay")

# --- 4. Lighting conditions (detector comparison) ---
LIGHTING_DIR = os.path.join(DATA_DIR, "lighting")   # subfolders: bright / normal / low_light

# --- 5. Head-pose conditions (detector comparison) ---
POSE_DIR = os.path.join(DATA_DIR, "pose")           # subfolders: frontal / left / right / up / down

os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_people(root_dir):
    """Returns sorted list of person-name subfolders under root_dir."""
    if not os.path.isdir(root_dir):
        return []
    return sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )


def list_images(folder):
    """Returns sorted list of full image paths in folder (non-recursive)."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    )


def check_data_present(*dirs, min_people=2, min_images_per_person=3):
    """
    Sanity check before running an expensive evaluation. Returns (ok, message).
    Doesn't crash the script — lets the caller print a friendly message and
    exit instead of a wall of DeepFace stack traces on an empty folder.
    """
    for d in dirs:
        people = list_people(d)
        if len(people) < min_people:
            return False, (
                f"Not enough data in {d}\n"
                f"  Found {len(people)} person folder(s), need at least {min_people}.\n"
                f"  Expected structure: {d}/<person_name>/*.jpg"
            )
        for p in people:
            imgs = list_images(os.path.join(d, p))
            if len(imgs) < min_images_per_person:
                return False, (
                    f"Person '{p}' in {d} only has {len(imgs)} image(s), "
                    f"need at least {min_images_per_person}."
                )
    return True, "ok"


def write_csv(path, rows, fieldnames):
    """rows: list of dicts. Writes header + rows."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {path}")


class Timer:
    """Tiny context manager for measuring elapsed ms."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
