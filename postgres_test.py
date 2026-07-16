#!/usr/bin/env python3
import os
import unittest

from backend import NeuralBlocksBackend
from mailer import MemoryMailer


POSTGRES_URL = os.environ.get("NBL_TEST_POSTGRES_URL")


@unittest.skipUnless(POSTGRES_URL, "NBL_TEST_POSTGRES_URL is not configured")
class PostgresBackendTest(unittest.TestCase):
    def setUp(self):
        import psycopg

        if os.environ.get("NBL_TEST_POSTGRES_RESET") == "1":
            with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
                connection.execute("DROP SCHEMA public CASCADE")
                connection.execute("CREATE SCHEMA public")
        self.backend = NeuralBlocksBackend(
            POSTGRES_URL,
            mailer=MemoryMailer(),
        )

    def test_postgres_registration_invitation_and_roster(self):
        registration = self.backend.register({
            "email": "postgres-admin@example.edu",
            "password": "postgres-admin-password",
            "displayName": "Postgres Admin",
            "createInstitution": True,
            "institutionName": "Postgres University",
            "institutionSlug": "postgres-university",
        })
        self.backend.verify_email({
            "token": registration["devVerificationToken"],
        })
        admin = self.backend.authenticate(registration["sessionToken"])
        course = self.backend.create_course(admin, {
            "name": "Database Systems",
            "code": "DB401",
            "term": "2026-2",
        })
        invitation = self.backend.create_invitation(admin, {
            "email": "postgres-professor@example.edu",
            "role": "professor",
            "courseId": course["id"],
        })
        accepted = self.backend.accept_invitation({
            "token": invitation["devInvitationToken"],
            "displayName": "Postgres Professor",
            "password": "postgres-professor-password",
        })
        self.assertEqual(accepted["user"]["role"], "professor")
        self.assertEqual(len(self.backend.list_course_members(admin, course["id"])), 2)
        self.assertEqual(self.backend.database.engine, "postgres")


if __name__ == "__main__":
    unittest.main()
