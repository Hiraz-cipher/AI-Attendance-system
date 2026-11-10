"""
manage_users.py
-----------------
Command-line tool to manage dashboard login accounts. Use this instead of
ever hardcoding usernames/passwords in app.py or auth.py.

Usage:
    python manage_users.py add <username> <password> <security|management>
    python manage_users.py list
    python manage_users.py delete <username>
    python manage_users.py passwd <username> <new_password>

Examples:
    python manage_users.py add jsmith Str0ngPass! security
    python manage_users.py add jdoe Str0ngPass! management
    python manage_users.py passwd admin NewSecurePassword2026
    python manage_users.py list
"""
import sys
import sqlite3
from werkzeug.security import generate_password_hash

from auth import init_users_table, DB_FILE, VALID_ROLES


def add_user(username, password, role):
    if role not in VALID_ROLES:
        print(f"Role must be one of: {', '.join(VALID_ROLES)}")
        return
    with sqlite3.connect(DB_FILE) as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, generate_password_hash(password), role),
            )
            print(f"Created user '{username}' with role '{role}'.")
        except sqlite3.IntegrityError:
            print(f"Username '{username}' already exists.")


def list_users():
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("SELECT username, role FROM users ORDER BY role, username").fetchall()
    if not rows:
        print("No users yet. Create one with: python manage_users.py add <username> <password> <role>")
        return
    for username, role in rows:
        print(f"  {username:<20} {role}")


def delete_user(username):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.execute("DELETE FROM users WHERE username=?", (username,))
        print(f"Deleted user '{username}'." if cur.rowcount else f"No such user '{username}'.")


def change_password(username, new_password):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash=? WHERE username=?",
            (generate_password_hash(new_password), username),
        )
        print(f"Password updated for '{username}'." if cur.rowcount else f"No such user '{username}'.")


def main():
    init_users_table()

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    if cmd == "add" and len(sys.argv) == 5:
        add_user(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "list":
        list_users()
    elif cmd == "delete" and len(sys.argv) == 3:
        delete_user(sys.argv[2])
    elif cmd == "passwd" and len(sys.argv) == 4:
        change_password(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
