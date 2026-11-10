import cv2
import pandas as pd
import os
import time
from datetime import datetime

from face_engine import FaceEngine
from liveness import LivenessDetector

# --- Configuration ---
DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"
MODEL_NAME = "VGG-Face"
RECOGNITION_INTERVAL = 2.0

if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)


def mark_attendance(name):
    """Marks attendance ensuring only one entry per person per day."""
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    now = datetime.now()
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

    already_logged = not df.empty and not df[(df["Name"] == name) & (df["Date"] == date_str)].empty
    if already_logged:
        return False

    new_record = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str}])
    pd.concat([df, new_record], ignore_index=True).to_csv(ATTENDANCE_FILE, index=False)
    print(f"[*] Attendance logged for {name} at {time_str}")
    return True


def get_camera():
    cam_idx_str = input("Enter camera index (0 for built-in, 1, 2 for external) [default 0]: ").strip()
    cam_idx = int(cam_idx_str) if cam_idx_str.isdigit() else 0
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    return cap


def main():
    cap = get_camera()
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    engine = FaceEngine(dataset_dir=DATASET_DIR, model_name=MODEL_NAME)
    liveness = LivenessDetector()

    print("Starting real-time recognition. Press 'q' to quit.")
    print("Blink naturally while facing the camera — attendance only logs after a live blink is detected.")
    last_recognition_time = 0
    marked_this_run = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            faces = engine.detect_faces(frame)
            live_status = liveness.process(frame)

            current_time = time.time()
            should_recognize = (current_time - last_recognition_time) > RECOGNITION_INTERVAL

            for (x, y, w, h) in faces:
                if should_recognize:
                    face_roi = frame[y:y + h, x:x + w]
                    name = engine.identify(face_roi)

                    if name and live_status["is_live"]:
                        cv2.putText(display_frame, f"{name} (Live)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if name not in marked_this_run:
                            mark_attendance(name)
                            marked_this_run.add(name)
                    elif name:
                        cv2.putText(display_frame, f"{name} - blink to verify", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 215, 255), 2)
                    else:
                        cv2.putText(display_frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if should_recognize:
                last_recognition_time = current_time

            cv2.imshow("Attendance Recognition", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        liveness.close()


if __name__ == "__main__":
    main()
