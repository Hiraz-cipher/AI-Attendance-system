import cv2
import os
import sys

def capture_dataset(student_name, cam_index=0, save_dir="dataset", target_size=(160, 160), max_images=10):
    """
    Captures a baseline facial dataset from the webcam.
    """
    # Create the directory for the student
    student_dir = os.path.join(save_dir, student_name)
    os.makedirs(student_dir, exist_ok=True)

    # Initialize Haar Cascade for face detection
    # Using Haar Cascade as it's very lightweight and suitable for CPU setups
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load Haar cascade classifier.")
        sys.exit(1)

    # Initialize video capture with the specified camera index
    # Using CAP_DSHOW on Windows often helps with external cameras
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index) # Fallback
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    # Attempt to set resolution to 1080p and framerate to 30fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"Starting dataset capture for {student_name}.")
    print("Press 'c' to capture an image (change head positions between captures).")
    print("Press 'q' to quit.")

    count = 0
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        display_frame = frame.copy()
        
        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Display progress and instructions
        cv2.putText(display_frame, f"Captured: {count}/{max_images}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Webcam - Dataset Collection", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Capture cancelled by user.")
            break
        elif key == ord('c'):
            if len(faces) == 0:
                print("No face detected. Please ensure your face is visible.")
            elif len(faces) > 1:
                print("Multiple faces detected. Please ensure only one face is in the frame.")
            else:
                # Exactly one face detected
                x, y, w, h = faces[0]
                
                # Crop the region of interest (ROI)
                roi = frame[y:y+h, x:x+w]
                
                # Resize to standard resolution (160x160 as requested)
                roi_resized = cv2.resize(roi, target_size)
                
                # Save the image to the structured local folder
                filename = os.path.join(student_dir, f"{student_name}_{count:02d}.jpg")
                cv2.imwrite(filename, roi_resized)
                print(f"Captured {filename}")
                count += 1
                
                # Provide visual feedback of capture
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow("Webcam - Dataset Collection", display_frame)
                cv2.waitKey(200) # Pause briefly to show capture flash effect

    if count == max_images:
        print(f"Successfully captured {max_images} images for {student_name}.")

    # Graceful Exit: Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter student name to start capturing: ").strip()
    if name:
        cam_idx_str = input("Enter camera index (0 for built-in, 1, 2, etc. for external) [default 0]: ").strip()
        cam_idx = int(cam_idx_str) if cam_idx_str.isdigit() else 0
        capture_dataset(name, cam_idx)
    else:
        print("Invalid name provided. Exiting.")
