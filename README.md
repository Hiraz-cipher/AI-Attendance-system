# AI-Based Attendance System

An ongoing academic prototype for recording attendance through face recognition, with blink/head-motion liveness checks and a Flask dashboard.

## Implemented features

- DeepFace/VGG-Face recognition with OpenCV face detection.
- MediaPipe facial landmarks for blink and head-motion checks.
- Management and security login roles with hashed passwords.
- Student registration, attendance logs, analytics, reports and eligibility pages.
- SQLite attendance storage with a once-per-person-per-day constraint.
- Camera configuration for local devices and supported phone camera streams.
- Optional background mirroring of attendance events to Firebase Firestore.

## Public source edition

This edition contains source code and setup instructions. Face photographs, biometric embeddings, account databases, attendance exports, Firebase credentials and evaluation datasets are excluded. The app starts with no registered students and no shared default accounts.

The evaluation scripts provide an evaluation workflow; the repository does not claim measured recognition accuracy or a validated anti-spoofing success rate. Blink/head-motion checks are a prototype and can be defeated by video replay.

## Local setup

Use a Python environment compatible with TensorFlow, DeepFace and MediaPipe. Python 3.11 is a starting point; the full dependency combination has not been validated on every platform.

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Then install dependencies, create your own local account, and start the app:

```sh
python -m pip install -r requirements.txt
python setup_account.py
python app.py
```

Open **http://localhost:5000** and sign in using the account you created. Choose the management role for full dashboard access. The app creates its local SQLite database. First use may download DeepFace weights and the MediaPipe landmark model. Fonts, icons and charts also use external CDNs.

## Camera and registration

1. Open **Phone camera** and select the device index or supported stream URL.
2. Verify the preview. The default index is 2, which may be different on your computer.
3. Register consenting test participants through **Add Student** using your management account.
4. Open **Mark Attendance**, then check the attendance log.

Phone Link support depends on Windows exposing the connected phone as a camera device. A phone connection alone does not guarantee a usable OpenCV camera. Physical camera operation and recognition speed must be tested locally.

## Configuration

- `CAM_SOURCE`: optional device index or stream URL override.
- `FLASK_SECRET_KEY`: set a private, stable random value if you need sessions to survive restarts. The default is generated per process.
- `FIREBASE_CREDENTIALS_PATH`: optional path to a local Firebase service-account JSON file. Install `firebase-admin` separately to enable cloud sync.

Keep credentials outside the repository. Without Firebase credentials, the app uses SQLite locally.

## Verification

Hardware-free camera checks:

```sh
python -m unittest discover -s tests -v
```

Recognition and liveness evaluations are documented in [evaluation/README.md](evaluation/README.md). Supply your own evaluation data; none is published here. Successful unit checks do not establish real-camera accuracy or production readiness.

## Project status

Ongoing academic prototype. Windows phone-camera operation, full ML inference, Firebase connectivity and end-to-end evaluation require testing in the target environment.
