"""
capture_eval_dataset.py
-------------------------
Captures images for the evaluation dataset — completely separate from the
production dataset/ folder your app.py uses. Run this once per volunteer.

Modes:
  1. Gallery + Probe (untrained nearest-neighbour recognition comparison)
     Captures N images of one person, splits them 80/20 into
     eval_data/recognition/gallery/<name>/ and
     eval_data/recognition/probe/<name>/.

  2. Classifier Train / Validation / Test (trained SVM & KNN)
     Captures N images of one person, splits them ~70/15/15 into
     eval_data/classifier/train/<name>/, .../validation/<name>/ and
     .../test/<name>/. This is a SEPARATE dataset from Gallery+Probe above
     — don't reuse the same session for both.

  3. Liveness — Genuine or Spoof (for TAR / FAR / FRR testing)
     Genuine: real person, natural blinking/movement, saved into
       eval_data/liveness/genuine/<name>/
     Spoof: printed photo or screen replay, saved into
       eval_data/liveness/spoof/printed_photo/<name>/ or
       eval_data/liveness/spoof/screen_replay/<name>/

  4. Lighting condition capture (for detector comparison)
     Saves images into eval_data/lighting/<condition>/<name>/
     where condition is one of: bright / normal / low_light

  5. Head-pose condition capture (for detector comparison)
     Saves images into eval_data/pose/<condition>/<name>/
     where condition is one of: frontal / left / right / up / down

Usage:
    python capture_eval_dataset.py
    (then follow the on-screen prompts)
"""

import os
import sys
import cv2

from eval_utils import (
    GALLERY_DIR, PROBE_DIR,
    CLASSIFIER_TRAIN_DIR, CLASSIFIER_VAL_DIR, CLASSIFIER_TEST_DIR,
    GENUINE_DIR, SPOOF_PHOTO_DIR, SPOOF_SCREEN_DIR,
    LIGHTING_DIR, POSE_DIR,
)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def get_camera(cam_idx):
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    return cap


def capture_n_images(save_dir, count, cam_idx, label_hint=""):
    if count <= 0:
        return 0

    os.makedirs(save_dir, exist_ok=True)
    existing = len([f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = get_camera(cam_idx)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return 0

    print(f"\nCapturing into: {save_dir}")
    if label_hint:
        print(f"({label_hint})")
    print("Press 'c' to capture, 'q' to stop early.\n")

    captured = 0
    while captured < count:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        display = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, f"Captured {captured}/{count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture - c to save, q to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if len(faces) != 1:
                print("Need exactly one face in frame.")
                continue
            x, y, w, h = faces[0]
            roi = cv2.resize(frame[y:y + h, x:x + w], (160, 160))
            idx = existing + captured
            filename = os.path.join(save_dir, f"img_{idx:03d}.jpg")
            cv2.imwrite(filename, roi)
            captured += 1
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    return captured


def do_gallery_probe(name, cam_idx):
    total = int(input("How many total images to capture for this person? [15]: ").strip() or 15)
    gallery_n = max(1, round(total * 0.8))
    probe_n = max(1, total - gallery_n)

    print(f"\nWill capture {gallery_n} images for GALLERY (registered set)")
    print("Vary angle and expression slightly between shots.")
    capture_n_images(os.path.join(GALLERY_DIR, name), gallery_n, cam_idx, "GALLERY set")

    print(f"\nNow {probe_n} images for PROBE (held-out test set) — vary conditions from the gallery shots.")
    capture_n_images(os.path.join(PROBE_DIR, name), probe_n, cam_idx, "PROBE set")

    print(f"\nDone. {name}: {gallery_n} gallery images, {probe_n} probe images.")


def do_classifier(name, cam_idx):
    print("\nThis builds the CLASSIFIER train/validation/test split — a separate")
    print("dataset from Gallery+Probe, used to actually TRAIN the SVM and KNN")
    print("classifiers (evaluate_recognition_svm.py / evaluate_recognition_knn.py).")
    total = int(input("How many total images to capture for this person? [20]: ").strip() or 20)

    train_n = max(1, round(total * 0.70))
    val_n = round(total * 0.15)
    test_n = max(1, total - train_n - val_n)

    print(f"\nSplit -> train: {train_n}, validation: {val_n}, test: {test_n}")

    print(f"\nCapturing {train_n} TRAIN images.")
    capture_n_images(os.path.join(CLASSIFIER_TRAIN_DIR, name), train_n, cam_idx, "CLASSIFIER TRAIN set")

    if val_n > 0:
        print(f"\nCapturing {val_n} VALIDATION images — vary conditions slightly from the train shots.")
        capture_n_images(os.path.join(CLASSIFIER_VAL_DIR, name), val_n, cam_idx, "CLASSIFIER VALIDATION set")

    print(f"\nCapturing {test_n} TEST images — ideally a different session/lighting than train.")
    capture_n_images(os.path.join(CLASSIFIER_TEST_DIR, name), test_n, cam_idx, "CLASSIFIER TEST set")

    print(f"\nDone. {name}: {train_n} train, {val_n} validation, {test_n} test images.")


def do_liveness(name, cam_idx):
    print("\nLiveness capture type:")
    print("  1. Genuine / live (real person — look straight, blink, move head naturally)")
    print("  2. Spoof - printed photo")
    print("  3. Spoof - screen replay (phone/tablet/laptop)")
    choice = input("Choose [1-3]: ").strip()

    if choice == "1":
        count = int(input("How many frames to capture? [30]: ").strip() or 30)
        print("Tip: talk, blink, and move your head naturally while capturing.")
        capture_n_images(os.path.join(GENUINE_DIR, name), count, cam_idx, "GENUINE (live) set")
    elif choice == "2":
        count = int(input("How many frames to capture? [10]: ").strip() or 10)
        capture_n_images(os.path.join(SPOOF_PHOTO_DIR, name), count, cam_idx, "SPOOF (printed photo) set")
    elif choice == "3":
        count = int(input("How many frames to capture? [10]: ").strip() or 10)
        capture_n_images(os.path.join(SPOOF_SCREEN_DIR, name), count, cam_idx, "SPOOF (screen replay) set")
    else:
        print("Invalid choice.")


def do_condition(name, cam_idx, kind):
    base = LIGHTING_DIR if kind == "lighting" else POSE_DIR
    if kind == "lighting":
        print("\nCondition — type one of: bright / normal / low_light")
    else:
        print("\nCondition — type one of: frontal / left / right / up / down")
    condition = input("Condition: ").strip().lower()
    count = int(input("How many images? [10]: ").strip() or 10)
    capture_n_images(os.path.join(base, condition, name), count, cam_idx, f"{kind} = {condition}")


def main():
    print("=== Evaluation Dataset Capture ===")
    print("This writes ONLY into evaluation/eval_data/ — your production")
    print("dataset/ folder used by app.py is never touched.\n")

    print("What are you capturing?")
    print("  1. Gallery + Probe Recognition Images (untrained nearest-neighbour comparison)")
    print("  2. Classifier Train/Validation/Test Images (trained SVM & KNN)")
    print("  3. Genuine + Spoof Liveness Images (TAR / FAR / FRR)")
    print("  4. Lighting Condition Images (bright / normal / low_light)")
    print("  5. Head-Pose Condition Images (frontal / left / right / up / down)")
    mode = input("Choose [1-5]: ").strip()

    name = input("Person name: ").strip()
    if not name:
        print("Name required.")
        sys.exit(1)
    name = "".join(c for c in name if c.isalnum() or c in " _-").strip().replace(" ", "_")

    cam_idx_str = input("Camera index [0]: ").strip()
    cam_idx = int(cam_idx_str) if cam_idx_str.isdigit() else 0

    if mode == "1":
        do_gallery_probe(name, cam_idx)
    elif mode == "2":
        do_classifier(name, cam_idx)
    elif mode == "3":
        do_liveness(name, cam_idx)
    elif mode == "4":
        do_condition(name, cam_idx, "lighting")
    elif mode == "5":
        do_condition(name, cam_idx, "pose")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
