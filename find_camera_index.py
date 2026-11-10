"""
find_camera_index.py
----------------------
Helper script — NOT part of the main app. Run this once to figure out which
numbered index (0, 1, 2...) corresponds to your phone camera vs your laptop's
built-in camera, so you can set CAM_SOURCE correctly in app.py.

Usage:
    python find_camera_index.py

It will open each detected camera index one at a time in a preview window,
labelled with its index number. Press any key to move to the next one.
Press 'q' at any point to stop early once you've spotted your phone.
"""

import cv2

MAX_INDEX_TO_CHECK = 5

print("Scanning camera indices 0 to", MAX_INDEX_TO_CHECK - 1, "...")
print("A preview window will pop up for each working camera.")
print("Look at the window title / image to identify which one is your phone.")
print("Press any key to move to the next camera, or 'q' to stop early.\n")

for idx in range(MAX_INDEX_TO_CHECK):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)

    if not cap.isOpened():
        print(f"Index {idx}: no camera found.")
        cap.release()
        continue

    ret, frame = cap.read()
    if not ret:
        print(f"Index {idx}: opened but couldn't read a frame.")
        cap.release()
        continue

    print(f"Index {idx}: camera found — showing preview. Press any key to continue.")
    cv2.putText(frame, f"CAMERA INDEX = {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(f"Camera index {idx} - press any key for next, q to quit", frame)

    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    cap.release()

    if key == ord('q'):
        break

print("\nDone. Use the index number of the correct camera as CAM_SOURCE in app.py.")
