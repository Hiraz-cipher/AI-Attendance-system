import cv2
import sqlite3
import csv
import io
import os
import re
import time
import secrets
import json
import threading
import atexit
import logging
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_file, request, session, redirect, url_for

from face_engine import FaceEngine
from liveness import LivenessDetector
from camera_stream import CameraStream
from auth import init_users_table, seed_default_users, verify_login, login_required, role_required
from firebase_sync import init_firebase, push_attendance_async

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
UI_VERSION = "2026.11.07-studio"


@app.after_request
def prevent_stale_pages(response):
    response.headers["X-Attendance-Version"] = UI_VERSION
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-store"
    return response


# Session cookies need a secret key to be signed. Falls back to a randomly
# generated key if FLASK_SECRET_KEY isn't set — which works fine for a single
# long-running process, but means everyone gets logged out if you restart the
# app without setting FLASK_SECRET_KEY explicitly. For anything beyond local
# testing, set a fixed value: set FLASK_SECRET_KEY=<random-string> (Windows)
# before running python app.py.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DB_FILE = "attendance.db"
MODEL_NAME = "VGG-Face"
RECOGNITION_INTERVAL = 2.0  # seconds between recognition attempts per stream

# Camera source: either an integer index for a locally plugged-in webcam (0, 1, 2...),
# or a string URL to stream from a phone camera app (e.g. IP Webcam on Android).
#   Local webcam example: CAM_SOURCE = 0
#   Phone stream example: CAM_SOURCE = "http://192.168.1.42:8080/video"
CAMERA_SETTINGS_FILE = os.path.join(BASE_DIR, "camera_settings.json")
try:
    with open(CAMERA_SETTINGS_FILE) as settings_file:
        camera_settings = json.load(settings_file)
except (OSError, ValueError):
    camera_settings = {"source": "2", "backend": "auto"}
CAM_SOURCE = os.environ.get("CAM_SOURCE", str(camera_settings.get("source", "2")))
CAMERA_CONFIG_LOCK = threading.Lock()

engine = FaceEngine(dataset_dir=DATASET_DIR, model_name=MODEL_NAME)

# One shared camera reader for the whole app — the dashboard's live feed and
# the registration page's preview both read from this instead of each
# opening their own connection to the camera (see camera_stream.py for why).
camera = CameraStream(CAM_SOURCE, backend=camera_settings.get("backend", "auto"))
atexit.register(camera.stop)


# --- Database ---
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                UNIQUE(name, date)
            )
            """
        )


init_db()
init_users_table()
seed_default_users()
init_firebase()


def mark_attendance(name):
    """Marks attendance. UNIQUE(name, date) constraint does the 'once per day'
    check at the database level instead of a separate SELECT-then-INSERT
    (fewer round trips, no race condition between the check and the insert).
    SQLite is the source of truth the dashboard reads from; a copy is also
    mirrored to Firestore in the background (no-op if cloud sync isn't set up)."""
    now = datetime.now()
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    with sqlite3.connect(DB_FILE) as conn:
        try:
            conn.execute(
                "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
                (name, date_str, time_str),
            )
            print(f"[*] Attendance logged for {name} at {time_str}")
            push_attendance_async(name, date_str, time_str)
        except sqlite3.IntegrityError:
            pass  # already logged today


def camera_placeholder(message="Camera unavailable - check connection and camera settings"):
    frame = np.full((480, 800, 3), (36, 27, 18), dtype=np.uint8)
    cv2.putText(frame, message, (24, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    ok, buffer = cv2.imencode(".jpg", frame)
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n" if ok else b""


@app.route("/api/camera/status")
@role_required("security", "management")
def camera_status():
    camera.start()
    return jsonify(ready=camera.is_open(), error=camera.error, source=str(camera.source))


@app.route("/camera-settings")
@role_required("security", "management")
def camera_settings_page():
    session.setdefault("camera_csrf", secrets.token_hex(24))
    return render_template("camera_settings.html", active_page="camera", camera_source=str(camera.source), camera_backend=camera.backend)


@app.route("/camera-preview")
@role_required("security", "management")
def camera_preview():
    def frames():
        camera.start()
        while True:
            frame = camera.get_frame()
            if frame is None:
                yield camera_placeholder("Connect your phone - open Phone camera settings")
            else:
                ok, buffer = cv2.imencode(".jpg", frame)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.05 if frame is not None else 0.5)
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/settings", methods=["POST"])
@role_required("security", "management")
def update_camera_settings():
    data = request.get_json(silent=True) or {}
    if not secrets.compare_digest(str(data.get("token", "")), session.get("camera_csrf", "missing")):
        return jsonify(error="Refresh this page and try again."), 403
    source, backend = str(data.get("source", "")), data.get("backend", "auto")
    if not source.isdecimal() or not 0 <= int(source) <= 10 or backend not in ("auto", "msmf", "dshow"):
        return jsonify(error="Choose a device number from 0 to 10 and a valid connection mode."), 400
    with CAMERA_CONFIG_LOCK:
        camera.stop()
        if camera._thread and camera._thread.is_alive():
            return jsonify(error="Camera driver is busy. Close other camera apps and restart this app."), 409
        camera.source, camera.backend = int(source), backend
        with open(CAMERA_SETTINGS_FILE, "w") as settings_file:
            json.dump({"source": source, "backend": backend}, settings_file)
        camera.start()
    return jsonify(ok=True)


# --- Video stream: Mark Attendance (dashboard live feed) ---
def generate_frames():
    # Fresh liveness state for each time the stream is opened.
    liveness = None
    camera.start()
    last_recognition_time = 0
    already_marked_this_session = set()  # avoid re-checking DB every single frame for the same person

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                yield camera_placeholder()
                time.sleep(0.5)
                continue

            if liveness is None:
                try:
                    liveness = LivenessDetector()
                except Exception:
                    logging.exception("Liveness initialization failed")
                    yield camera_placeholder("Verification unavailable. Check model and dependencies.")
                    time.sleep(2)
                    continue
            display_frame = frame.copy()
            faces = engine.detect_faces(frame)

            # Run liveness (blink + head-pose tracking) every frame, independent
            # of the recognition cadence — blinks are brief, so we can't afford
            # to only sample liveness once every 2 seconds or we'd miss most of them.
            live_status = liveness.process(frame)

            current_time = time.time()
            should_recognize = (current_time - last_recognition_time) > RECOGNITION_INTERVAL

            # A single liveness result must never verify several people.
            if len(faces) != 1:
                live_status["is_live"] = False
            for (x, y, w, h) in faces:
                if should_recognize:
                    face_roi = frame[y:y + h, x:x + w]
                    name = engine.identify(face_roi)

                    if name and live_status["is_live"]:
                        # Recognized AND passed the liveness check
                        cv2.putText(display_frame, f"{name} (Live)", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if name not in already_marked_this_session:
                            mark_attendance(name)
                            already_marked_this_session.add(name)
                    elif name:
                        # Recognized but no blink/head-movement seen recently — don't
                        # mark attendance yet; catches someone holding up a static photo.
                        cv2.putText(display_frame, f"{name} - blink to verify", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 215, 255), 2)
                    else:
                        cv2.putText(display_frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    # Still show a live box between recognition cycles, just skip the expensive call
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 150, 0), 2)

            if should_recognize:
                last_recognition_time = current_time

            ret, buffer = cv2.imencode(".jpg", display_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        if liveness is not None:
            liveness.close()


# --- Video stream: Add Student (registration preview, no recognition/marking) ---
def generate_register_frames():
    camera.start()
    while True:
        frame = camera.get_frame()
        if frame is None:
            yield camera_placeholder()
            time.sleep(0.5)
            continue

        display_frame = frame.copy()
        faces = engine.detect_faces(frame)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Ready to capture", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif len(faces) > 1:
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(display_frame, "Only one person at a time", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "No face detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        ret, buffer = cv2.imencode(".jpg", display_frame)
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# --- Auth ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role = verify_login(username, password)

        if role:
            session["username"] = username
            session["role"] = role
            next_url = request.args.get("next")
            if next_url:
                return redirect(next_url)
            return redirect(url_for("mark_attendance_page") if role == "security" else url_for("index"))

        return render_template("login.html", error="Incorrect username or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/mark-attendance")
@role_required("security", "management")
def mark_attendance_page():
    # Security gets a stripped-down, sidebar-free page (mark_attendance.html)
    # so they can't see or navigate to pages they don't have access to.
    # Management gets the same live feed inside the full dashboard layout.
    if session.get("role") == "management":
        return render_template("mark_attendance_admin.html", active_page="mark")
    return render_template("mark_attendance.html")


@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403


# --- Routes ---
@app.route("/")
@role_required("management")
def index():
    return render_template("overview.html", active_page="overview")


@app.route("/log")
@role_required("management")
def log_page():
    return render_template("log.html", active_page="log")


@app.route("/analytics")
@role_required("management")
def analytics_page():
    return render_template("analytics.html", active_page="analytics")


@app.route("/eligibility")
@role_required("management")
def eligibility_page():
    return render_template("eligibility.html", active_page="eligibility")


@app.route("/reports")
@role_required("management")
def reports_page():
    return render_template("reports.html", active_page="reports")


@app.route("/video_feed")
@role_required("security", "management")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# --- Add Student (registration) ---
def _safe_student_dirname(name):
    """Turns a display name into a safe folder name: letters, numbers,
    spaces and underscores only, collapsed and trimmed. Prevents path
    traversal (e.g. someone typing '../../etc') and filesystem-illegal
    characters from ever reaching os.path.join."""
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9 _-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


@app.route("/register")
@role_required("management")
def register_page():
    return render_template("register.html", active_page="register")


@app.route("/register_feed")
@role_required("management")
def register_feed():
    return Response(generate_register_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/register/capture", methods=["POST"])
@role_required("management")
def register_capture():
    data = request.get_json(silent=True) or {}
    raw_name = data.get("name", "")
    name = _safe_student_dirname(raw_name)

    if not name:
        return jsonify({"ok": False, "error": "Please enter a valid student name."}), 400

    frame = camera.get_frame()
    if frame is None:
        return jsonify({"ok": False, "error": "Camera not ready yet — try again in a moment."}), 503

    faces = engine.detect_faces(frame)
    if len(faces) == 0:
        return jsonify({"ok": False, "error": "No face detected. Make sure your face is clearly visible."}), 422
    if len(faces) > 1:
        return jsonify({"ok": False, "error": "Multiple faces detected. Only one person should be in frame."}), 422

    x, y, w, h = faces[0]
    roi = frame[y:y + h, x:x + w]
    roi_resized = cv2.resize(roi, (160, 160))

    student_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(student_dir, exist_ok=True)

    existing = [f for f in os.listdir(student_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    next_index = len(existing)
    filename = os.path.join(student_dir, f"{name}_{next_index:02d}.jpg")
    cv2.imwrite(filename, roi_resized)

    return jsonify({"ok": True, "count": next_index + 1})


@app.route("/api/register/finish", methods=["POST"])
@role_required("management")
def register_finish():
    data = request.get_json(silent=True) or {}
    name = _safe_student_dirname(data.get("name", ""))
    student_dir = os.path.join(DATASET_DIR, name)

    photo_count = 0
    if os.path.isdir(student_dir):
        photo_count = len([f for f in os.listdir(student_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if photo_count == 0:
        return jsonify({"ok": False, "error": "No photos were captured for this student yet."}), 400

    # Rebuild DeepFace's cached embeddings so this new student is recognizable
    # immediately, without needing to restart the app or run encode_faces.py by hand.
    engine.rebuild_embeddings()

    return jsonify({"ok": True, "photo_count": photo_count})


@app.route("/api/attendance")
@role_required("management")
def get_attendance():
    """Attendance log rows. Accepts ?date=YYYY-MM-DD (defaults to today) so
    the Attendance Log page can look back at any past day, not just today."""
    date_str = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            "SELECT name, time FROM attendance WHERE date=? ORDER BY time DESC", (date_str,)
        ).fetchall()
    return jsonify([{"name": r[0], "time": r[1]} for r in rows])


@app.route("/api/attendance/summary")
@role_required("management")
def get_summary():
    """Per-person total attendance days — feeds a chart on the dashboard (Layer 7)."""
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            "SELECT name, COUNT(DISTINCT date) as days FROM attendance GROUP BY name ORDER BY days DESC"
        ).fetchall()
    return jsonify([{"name": r[0], "days": r[1]} for r in rows])


@app.route("/api/attendance/stats")
@role_required("management")
def get_stats():
    """KPI numbers for the dashboard home: total registered, present/absent,
    and attendance percentage — matches the project spec's Dashboard Home
    requirements. Accepts ?range=today|week|month (default today), driven by
    the Today/This Week/This Month pill tabs on the dashboard.
    'Present' means at least one verified check-in somewhere in that window,
    not present on every single day of it."""
    total_registered = 0
    if os.path.isdir(DATASET_DIR):
        total_registered = sum(
            1 for entry in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, entry))
        )

    range_param = request.args.get("range", "today")
    if range_param == "week":
        since_clause = "date >= date('now', '-6 days')"
    elif range_param == "month":
        since_clause = "date >= date('now', '-29 days')"
    else:
        range_param = "today"
        since_clause = "date = date('now')"

    with sqlite3.connect(DB_FILE) as conn:
        present = conn.execute(
            f"SELECT COUNT(DISTINCT name) FROM attendance WHERE {since_clause}"
        ).fetchone()[0]

    absent = max(total_registered - present, 0)
    percentage = round((present / total_registered) * 100, 1) if total_registered else 0.0

    return jsonify({
        "range": range_param,
        "total_registered": total_registered,
        "present_today": present,
        "absent_today": absent,
        "attendance_percentage": percentage,
    })


@app.route("/api/attendance/trends")
@role_required("management")
def get_trends():
    """Daily present-count for the last 14 days — feeds the weekly/monthly
    trend chart the spec calls for under Analytics."""
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            """
            SELECT date, COUNT(DISTINCT name) as present
            FROM attendance
            WHERE date >= date('now', '-13 days')
            GROUP BY date
            ORDER BY date ASC
            """
        ).fetchall()
    return jsonify([{"date": r[0], "present": r[1]} for r in rows])


def _compute_eligibility(threshold):
    """Shared by the JSON endpoint and the CSV export so both always agree.
    'Attendance %' here = (days this student was marked present) / (total
    distinct days ANYONE was marked present, i.e. how many sessions the
    system has actually recorded) * 100. Registered students who never
    attended a single session still appear, at 0%."""
    all_students = []
    if os.path.isdir(DATASET_DIR):
        all_students = sorted(
            entry for entry in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, entry))
        )

    with sqlite3.connect(DB_FILE) as conn:
        total_days = conn.execute("SELECT COUNT(DISTINCT date) FROM attendance").fetchone()[0]
        present_by_name = dict(conn.execute(
            "SELECT name, COUNT(DISTINCT date) FROM attendance GROUP BY name"
        ).fetchall())

    results = []
    for name in all_students:
        present_days = present_by_name.get(name, 0)
        percentage = round((present_days / total_days) * 100, 1) if total_days else 0.0
        results.append({
            "name": name,
            "present_days": present_days,
            "total_days": total_days,
            "percentage": percentage,
            "eligible": percentage >= threshold,
        })

    results.sort(key=lambda r: r["percentage"], reverse=True)
    return results, total_days


@app.route("/api/attendance/eligibility")
@role_required("management")
def get_eligibility():
    """Exam-eligibility list: every registered student's attendance %, and
    whether they clear the given threshold (?threshold=80, default 80)."""
    threshold = request.args.get("threshold", 80, type=float)
    results, total_days = _compute_eligibility(threshold)
    return jsonify({"threshold": threshold, "total_days": total_days, "students": results})


@app.route("/api/attendance/eligibility/export")
@role_required("management")
def export_eligibility():
    """CSV of the eligibility list — for handing to the exams office."""
    threshold = request.args.get("threshold", 80, type=float)
    results, total_days = _compute_eligibility(threshold)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Name", "Present Days", "Total Days", "Attendance %", f"Eligible (>= {threshold}%)"])
    for r in results:
        writer.writerow([r["name"], r["present_days"], r["total_days"], r["percentage"], "Yes" if r["eligible"] else "No"])
    buf.seek(0)

    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    filename = f"exam_eligibility_{int(threshold)}pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=filename)


@app.route("/api/attendance/weekday-pattern")
@role_required("management")
def get_weekday_pattern():
    """Average number of students present, broken down by day of the week —
    surfaces patterns like 'Mondays consistently run low', which the raw
    14-day trend line doesn't show on its own."""
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("SELECT date, COUNT(DISTINCT name) FROM attendance GROUP BY date").fetchall()

    from collections import defaultdict
    by_weekday = defaultdict(list)
    for date_str, present_count in rows:
        weekday_index = datetime.strptime(date_str, "%Y-%m-%d").weekday()  # 0=Monday
        by_weekday[weekday_index].append(present_count)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    result = []
    for i, label in enumerate(day_names):
        values = by_weekday.get(i, [])
        avg = round(sum(values) / len(values), 1) if values else 0.0
        result.append({"day": label, "average_present": avg, "sessions_recorded": len(values)})

    return jsonify(result)


@app.route("/api/attendance/export")
@role_required("management")
def export_attendance():
    """CSV export of attendance records — the 'Export Reports' box in Layer 7.
    Optional ?start=YYYY-MM-DD&end=YYYY-MM-DD narrows the export to a date
    range; omitted, it exports the full history (unchanged default behavior)."""
    start = request.args.get("start")
    end = request.args.get("end")

    query = "SELECT name, date, time FROM attendance"
    params = []
    if start and end:
        query += " WHERE date BETWEEN ? AND ?"
        params = [start, end]
    elif start:
        query += " WHERE date >= ?"
        params = [start]
    elif end:
        query += " WHERE date <= ?"
        params = [end]
    query += " ORDER BY date DESC, time DESC"

    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(query, params).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Name", "Date", "Time"])
    writer.writerows(rows)
    buf.seek(0)

    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    filename = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=filename)


if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", "5000"))
        print(f"\nAttendance UI {UI_VERSION}\nProject folder: {BASE_DIR}\nOpen: http://localhost:{port}\n", flush=True)
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    finally:
        camera.stop()
