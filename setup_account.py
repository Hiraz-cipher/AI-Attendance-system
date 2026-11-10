"""Create a local account without placing its password in shell history."""
from getpass import getpass
from auth import init_users_table, VALID_ROLES
from manage_users import add_user


def main():
    init_users_table()
    username = input("Username: ").strip()
    if not username:
        raise SystemExit("Username is required.")
    role = input("Role (management/security): ").strip()
    if role not in VALID_ROLES:
        raise SystemExit("Choose management or security.")
    password = getpass("Password: ")
    if not password:
        raise SystemExit("Password cannot be empty.")
    if password != getpass("Confirm password: "):
        raise SystemExit("Passwords do not match.")
    add_user(username, password, role)


if __name__ == "__main__":
    main()
