"""
evaluate_liveness.py
----------------------
Evaluates your ACTUAL LivenessDetector (imported directly from the parent
project's liveness.py — not reimplemented here) against live footage vs.
printed-photo and screen-replay spoof attempts (Section 16 "Liveness
Evaluation" + your proposal's <2% FAR target).

Because it imports the real class, any tuning you do in liveness.py
(thresholds, LIVE_WINDOW_SECONDS, etc.) is automatically reflected here —
this evaluation always measures what your app actually runs.

Input format: each "sample" is a short image sequence (a folder of frames)
or a video file, representing one continuous attempt:
  eval_data/liveness/genuine/<name>/              (frames or a .mp4)
  eval_data/liveness/spoof/printed_photo/<name>/   (frames or a .mp4)
  eval_data/liveness/spoof/screen_replay/<name>/   (frames or a .mp4)

For GENUINE samples, capture ~10-15 seconds of normal webcam footage per
volunteer (talking, blinking naturally) either as a video file
(genuine/<name>/clip.mp4) or an image-sequence folder of frames pulled
from that footage. Use capture_eval_dataset.py mode 3 ("Genuine + Spoof
Liveness Images") to build this whole set.

A sample is scored "accepted" (is_live=True) if the detector reports
is_live=True at ANY point while processing the sequence — matching how the
real app gates recognition frame-by-frame, not just on the final frame.

Output: results/liveness_results.csv, results/liveness_summary.csv
"""

import os
import glob

import cv2

from eval_utils import RESULTS_DIR, write_csv, list_people, list_images
from eval_utils import GENUINE_DIR, SPOOF_PHOTO_DIR, SPOOF_SCREEN_DIR

VIDEO_EXTS = (".mp4", ".avi", ".mov")


def find_samples(root_dir):
    """
    root_dir/<person>/  containing either:
      - one or more video files, each treated as a separate sample, OR
      - a flat folder of image frames, treated as ONE sample (a sequence)
    Returns list of (person, sample_id, frame_source) where frame_source is
    either a video path (str) or a list of image paths.
    """
    samples = []
    for person in list_people(root_dir):
        person_dir = os.path.join(root_dir, person)
        videos = [f for ext in VIDEO_EXTS for f in glob.glob(os.path.join(person_dir, f"*{ext}"))]
        if videos:
            for v in videos:
                samples.append((person, os.path.basename(v), v))
        else:
            imgs = list_images(person_dir)
            if imgs:
                samples.append((person, "frame_sequence", imgs))
    return samples


def frames_from_source(source):
    """Yields BGR frames from either a video path or a list of image paths."""
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()
    else:
        for img_path in source:
            frame = cv2.imread(img_path)
            if frame is not None:
                yield frame


def run_sample(source):
    """
    Runs the real LivenessDetector over every frame in the sample.
    Returns (accepted_at_any_point: bool, total_frames: int,
             total_blinks: int, total_movements: int).
    """
    from liveness import LivenessDetector  # the actual production class
    detector = LivenessDetector()
    accepted = False
    n_frames = 0
    last_status = None
    try:
        for frame in frames_from_source(source):
            status = detector.process(frame)
            last_status = status
            n_frames += 1
            if status["is_live"]:
                accepted = True
    finally:
        detector.close()

    total_blinks = last_status["total_blinks"] if last_status else 0
    total_movements = last_status["total_movements"] if last_status else 0
    return accepted, n_frames, total_blinks, total_movements


def evaluate_category(root_dir, category_label):
    samples = find_samples(root_dir)
    rows = []
    if not samples:
        print(f"  No samples found in {root_dir}")
        return rows

    for person, sample_id, source in samples:
        print(f"  Processing {category_label}/{person}/{sample_id} ...")
        accepted, n_frames, blinks, movements = run_sample(source)
        rows.append({
            "category": category_label,
            "person": person,
            "sample": sample_id,
            "n_frames": n_frames,
            "accepted_as_live": accepted,
            "total_blinks": blinks,
            "total_movements": movements,
        })
        print(f"    -> accepted_as_live={accepted}  (blinks={blinks}, movements={movements}, frames={n_frames})")
    return rows


def main():
    print("=== Liveness / Anti-Spoofing Evaluation ===")
    print("(uses your actual liveness.py LivenessDetector, unmodified)\n")

    all_rows = []
    print("-- GENUINE samples (real people) --")
    all_rows += evaluate_category(GENUINE_DIR, "genuine")
    print("\n-- SPOOF: printed photo --")
    all_rows += evaluate_category(SPOOF_PHOTO_DIR, "spoof_printed_photo")
    print("\n-- SPOOF: screen replay --")
    all_rows += evaluate_category(SPOOF_SCREEN_DIR, "spoof_screen_replay")

    if not all_rows:
        print("\nNo data found at all. Expected:")
        print(f"  {GENUINE_DIR}/<person>/(clip.mp4 or frames)")
        print(f"  {SPOOF_PHOTO_DIR}/<person>/(clip.mp4 or frames)")
        print(f"  {SPOOF_SCREEN_DIR}/<person>/(clip.mp4 or frames)")
        print("\nUse capture_eval_dataset.py (mode 3) to build the whole liveness set —")
        print("choose 'Genuine / live' for genuine, or one of the spoof options for spoof.")
        return

    write_csv(
        os.path.join(RESULTS_DIR, "liveness_results.csv"),
        all_rows,
        fieldnames=["category", "person", "sample", "n_frames",
                    "accepted_as_live", "total_blinks", "total_movements"],
    )

    # --- Summary: TAR / FAR / FRR ---
    def rate(rows, want_accept):
        if not rows:
            return None
        matched = sum(1 for r in rows if r["accepted_as_live"] == want_accept)
        return round(100 * matched / len(rows), 2)

    genuine_rows = [r for r in all_rows if r["category"] == "genuine"]
    spoof_photo_rows = [r for r in all_rows if r["category"] == "spoof_printed_photo"]
    spoof_screen_rows = [r for r in all_rows if r["category"] == "spoof_screen_replay"]
    all_spoof_rows = spoof_photo_rows + spoof_screen_rows

    summary = [
        {"metric": "TAR (True Accept Rate, genuine correctly accepted)", "value_pct": rate(genuine_rows, True), "n_samples": len(genuine_rows)},
        {"metric": "FRR (False Reject Rate, genuine incorrectly rejected)", "value_pct": rate(genuine_rows, False), "n_samples": len(genuine_rows)},
        {"metric": "FAR overall (spoof incorrectly accepted)", "value_pct": rate(all_spoof_rows, True), "n_samples": len(all_spoof_rows)},
        {"metric": "FAR - printed photo only", "value_pct": rate(spoof_photo_rows, True), "n_samples": len(spoof_photo_rows)},
        {"metric": "FAR - screen replay only", "value_pct": rate(spoof_screen_rows, True), "n_samples": len(spoof_screen_rows)},
    ]

    write_csv(
        os.path.join(RESULTS_DIR, "liveness_summary.csv"),
        summary,
        fieldnames=["metric", "value_pct", "n_samples"],
    )

    print("\n=== Summary ===")
    for row in summary:
        print(f"{row['metric']}: {row['value_pct']}%  (n={row['n_samples']})")

    far_overall = rate(all_spoof_rows, True)
    if far_overall is not None:
        target_note = "MEETS <2% target" if far_overall < 2 else "ABOVE 2% target — worth discussing in report"
        print(f"\nProposal target: FAR < 2%. Measured overall FAR: {far_overall}% -> {target_note}")


if __name__ == "__main__":
    main()
