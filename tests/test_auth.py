"""Local authentication checks; no camera or ML dependencies are required."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import auth
import manage_users


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = str(Path(self.temp.name) / 'test.db')
        for module in (auth, manage_users):
            patcher = patch.object(module, 'DB_FILE', path)
            patcher.start()
            self.addCleanup(patcher.stop)
        auth.init_users_table()

    def test_no_shared_default_accounts(self):
        auth.seed_default_users()
        with sqlite3.connect(auth.DB_FILE) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM users').fetchone()[0], 0)

    def test_account_is_hashed_and_login_checks_password(self):
        manage_users.add_user('test-user', 'test-only-password', 'management')
        self.assertEqual(auth.verify_login('test-user', 'test-only-password'), 'management')
        self.assertIsNone(auth.verify_login('test-user', 'wrong-password'))
        self.assertIsNone(auth.verify_login('missing-user', 'test-only-password'))
        with sqlite3.connect(auth.DB_FILE) as connection:
            stored = connection.execute('SELECT password_hash FROM users').fetchone()[0]
        self.assertNotEqual(stored, 'test-only-password')
        auth.seed_default_users()
        self.assertEqual(auth.verify_login('test-user', 'test-only-password'), 'management')

    def test_invalid_role_does_not_create_an_account(self):
        manage_users.add_user('test-user', 'test-only-password', 'invalid-role')
        with sqlite3.connect(auth.DB_FILE) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM users').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
