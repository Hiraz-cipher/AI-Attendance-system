"""
auth.py
--------
Login and role-based access control for the dashboard.

Two roles, matching the two login types requested:
  - "security"   -> gate staff. Can Mark Attendance and Add Student only.
  - "management" -> full dashboard: KPIs, analytics, attendance log, reports/export,
                     plus everything security can do.

Passwords are hashed with werkzeug's generate_password_hash/check_password_hash
(bundled with Flask already — nothing new to install). Plaintext passwords are
never stored.

Don't hardcode real accounts in this file or in app.py — use manage_users.py
to create/change accounts instead, so credentials never end up committed to
your project files or shared alongside your code.
"""

import sqlite3
from functools import wraps
from flask import session, redirect, url_for, request, abort
from werkzeug.security import generate_password_hash, check_password_hash

DB_FILE = "attendance.db"
VALID_ROLES = ("security", "management")


def init_users_table():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('security','management'))
            )
            """
        )


def seed_default_users():
    """Retained startup hook; public source never creates shared login accounts."""
    with sqlite3.connect(DB_FILE) as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        print("[auth] No accounts yet. Run python setup_account.py to create one.")


def verify_login(username, password):
    """Returns the user's role if credentials are correct, else None."""
    with sqlite3.connect(DB_FILE) as conn:
        row = conn.execute(
            "SELECT password_hash, role FROM users WHERE username=?", (username,)
        ).fetchone()
    if row and check_password_hash(row[0], password):
        return row[1]
    return None


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapper


def role_required(*roles):
    """Restricts a route to one or more roles. Usage:
        @role_required("management")
        @role_required("security", "management")
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "username" not in session:
                return redirect(url_for("login", next=request.path))
            if session.get("role") not in roles:
                abort(403)
            return f(*args, **kwargs)
        return wrapper
    return decorator
