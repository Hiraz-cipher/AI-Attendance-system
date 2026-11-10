"""
firebase_sync.py
------------------
Optional Firebase Firestore mirror for attendance records — satisfies the
project spec's "cloud database" requirement (Firebase/AWS RDS) without
replacing SQLite as the dashboard's primary store.

Why a mirror instead of a full migration: the dashboard runs several SQL
aggregations per page load (COUNT/GROUP BY for KPI cards, trend charts,
weekday patterns, exam eligibility). Firestore's NoSQL document model
doesn't support those query shapes natively — replicating them would mean
either restructuring the entire data model around Firestore, or pulling
every record to the server and aggregating in Python on every request
(slow, and defeats the point of a managed cloud database). Mirroring keeps
SQLite as the fast local source of truth the dashboard actually reads from,
and pushes a copy of every attendance event to Firestore in the background
— a real, remotely-accessible cloud record, which is what the spec is
asking for.

If the credentials file isn't present, cloud sync silently disables itself
and the app keeps working exactly as before on SQLite alone — this file is
designed to never be the reason the app breaks.
"""

import os
import threading

FIREBASE_ENABLED = False
_db = None
_init_lock = threading.Lock()


def init_firebase():
    """Call once at app startup. Safe to call even if firebase-admin isn't
    installed or no credentials file exists — just disables cloud sync."""
    global FIREBASE_ENABLED, _db

    cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")

    if not os.path.exists(cred_path):
        print(f"[firebase] No credentials file found at '{cred_path}' — cloud sync disabled.")
        print("[firebase] The app works fully without it (SQLite only). See README for setup steps.")
        return

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        _db = firestore.client()
        FIREBASE_ENABLED = True
        print("[firebase] Cloud sync enabled — attendance records will mirror to Firestore.")
    except ImportError:
        print("[firebase] firebase-admin isn't installed — run: pip install firebase-admin")
        print("[firebase] Cloud sync disabled — the app works fully without it.")
    except Exception as e:
        print(f"[firebase] Failed to initialize ({e}) — cloud sync disabled.")
        print("[firebase] The app works fully without it (SQLite only).")


def push_attendance_async(name, date_str, time_str):
    """Fire-and-forget push to Firestore. Runs on a background thread and
    never raises back into the caller — a Firestore/network hiccup must
    never break local attendance marking, which is why this doesn't return
    anything the caller has to check. SQLite remains the source of truth;
    this is a best-effort mirror on top of it."""
    if not FIREBASE_ENABLED:
        return

    def _push():
        try:
            doc_id = f"{name}_{date_str}"  # matches SQLite's one-row-per-person-per-day
            _db.collection("attendance").document(doc_id).set({
                "name": name,
                "date": date_str,
                "time": time_str,
            })
        except Exception as e:
            # Not re-raised: this runs on a background thread with no caller
            # waiting on it, so printing is the only way to surface the issue.
            print(f"[firebase] Sync failed for {name} on {date_str} ({e}) — local record is unaffected.")

    threading.Thread(target=_push, daemon=True).start()
