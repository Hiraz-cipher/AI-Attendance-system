"""
evaluate_detectors.py
-----------------------
Compares face detector backends against your captured lighting_test/ and
pose_test/ image sets (Section 16 "Face Detector Comparison" in the
proposal).

Backends compared:
  - "opencv"  -> Haar Cascade (what your production face_engine.py uses today)
  - "yolov8"  -> YOLOv8 face detector (the alternative your proposal names)
  - You can add more by editing DETECTOR_BACKENDS below (e.g. "retinaface",
    "mtcnn", "mediapipe") if you want a wider comparison.

Uses DeepFace.extract_faces() as a common interface across backends instead
of hand-rolling separate detection code per model — this keeps the
comparison apples-to-apples (same input handling, same face-count logic)
and doesn't require you to install/wire up YOLOv8 separately; DeepFace
downloads and manages the backend weights itself on first use.

Requires: eval_data/lighting/<condition>/<person>/*.jpg
            (condition = bright / normal / low_light)
          eval_data/pose/<condition>/<person>/*.jpg
            (condition = frontal / left / right / up / down)
Each image should contain exactly ONE face (use capture_eval_dataset.py
mode 4/5 to build this).

Output: results/detector_comparison.csv, results/detector_summary.csv
"""

import os
import time
import traceback

from eval_utils import LIGHTING_DIR, POSE_DIR, RESULTS_DIR, write_csv, list_images

DETECTOR_BACKENDS = ["opencv", "yolov8"]  # add "retinaface", "mtcnn", "mediapipe" etc. if desired


def gather_condition_images(root_dir):
    """
    root_dir/<condition>/<person>/*.jpg
    Returns list of (condition, image_path).
    """
    items = []
    if not os.path.isdir(root_dir):
        return items
    for condition in sorted(os.listdir(root_dir)):
        cond_dir = os.path.join(root_dir, condition)
        if not os.path.isdir(cond_dir):
            continue
        for person in sorted(os.listdir(cond_dir)):
            person_dir = os.path.join(cond_dir, person)
            if not os.path.isdir(person_dir):
                continue
            for img_path in list_images(person_dir):
                items.append((condition, img_path))
    return items


def run_detector(backend, image_path):
    """
    Returns (num_faces_found, elapsed_ms, error_or_None).
    """
    from deepface import DeepFace
    start = time.perf_counter()
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=backend,
            enforce_detection=False,
            align=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # DeepFace returns a list even when nothing meaningful was found;
        # confidence field lets us filter out near-zero "detections".
        real_faces = [f for f in faces if f.get("confidence", 1) > 0]
        return len(real_faces), elapsed_ms, None
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return 0, elapsed_ms, str(e)


def main():
    print("=== Face Detector Comparison ===\n")

    all_items = []
    all_items += [("lighting", cond, path) for cond, path in gather_condition_images(LIGHTING_DIR)]
    all_items += [("pose", cond, path) for cond, path in gather_condition_images(POSE_DIR)]

    if not all_items:
        print("No data found. Expected images under:")
        print(f"  {LIGHTING_DIR}/<condition>/<person>/*.jpg")
        print(f"  {POSE_DIR}/<condition>/<person>/*.jpg")
        print("\nUse capture_eval_dataset.py (modes 4 and 5) to collect these first.")
        return

    print(f"Found {len(all_items)} test images across "
          f"{len(set(c for _, c, _ in all_items))} conditions.\n")

    detailed_rows = []
    for backend in DETECTOR_BACKENDS:
        print(f"--- Testing backend: {backend} ---")
        for test_type, condition, img_path in all_items:
            n_faces, elapsed_ms, error = run_detector(backend, img_path)
            correct = (n_faces == 1)  # each eval image is expected to contain exactly one face
            detailed_rows.append({
                "backend": backend,
                "test_type": test_type,
                "condition": condition,
                "image": os.path.basename(img_path),
                "faces_found": n_faces,
                "correct_detection": correct,
                "latency_ms": round(elapsed_ms, 2),
                "error": error or "",
            })
            status = "OK" if correct else ("MISS" if n_faces == 0 else "MULTI")
            print(f"  [{status}] {condition}/{os.path.basename(img_path)} "
                  f"({n_faces} face(s), {elapsed_ms:.1f}ms)")

    write_csv(
        os.path.join(RESULTS_DIR, "detector_comparison.csv"),
        detailed_rows,
        fieldnames=["backend", "test_type", "condition", "image",
                    "faces_found", "correct_detection", "latency_ms", "error"],
    )

    # --- Summary per backend / condition ---
    summary_rows = []
    for backend in DETECTOR_BACKENDS:
        backend_rows = [r for r in detailed_rows if r["backend"] == backend]
        for test_type in ["lighting", "pose"]:
            conditions = sorted(set(r["condition"] for r in backend_rows if r["test_type"] == test_type))
            for condition in conditions:
                subset = [r for r in backend_rows if r["test_type"] == test_type and r["condition"] == condition]
                if not subset:
                    continue
                n = len(subset)
                correct = sum(1 for r in subset if r["correct_detection"])
                avg_latency = sum(r["latency_ms"] for r in subset) / n
                summary_rows.append({
                    "backend": backend,
                    "test_type": test_type,
                    "condition": condition,
                    "n_images": n,
                    "detection_rate_pct": round(100 * correct / n, 1),
                    "avg_latency_ms": round(avg_latency, 2),
                    "avg_fps": round(1000.0 / avg_latency, 2) if avg_latency > 0 else 0,
                })

    write_csv(
        os.path.join(RESULTS_DIR, "detector_summary.csv"),
        summary_rows,
        fieldnames=["backend", "test_type", "condition", "n_images",
                    "detection_rate_pct", "avg_latency_ms", "avg_fps"],
    )

    print("\n=== Summary ===")
    for row in summary_rows:
        print(f"{row['backend']:10s} | {row['test_type']:8s} | {row['condition']:10s} | "
              f"detect={row['detection_rate_pct']}% | {row['avg_fps']} FPS")


if __name__ == "__main__":
    main()
