#!/usr/bin/env python3
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from migrate import sqlalchemy_url


ROOT = Path(__file__).resolve().parent


class MigrationTest(unittest.TestCase):
    def test_postgres_url_uses_installed_psycopg_driver(self):
        self.assertEqual(
            sqlalchemy_url("postgresql://user:pass@db:5432/neural_blocks"),
            "postgresql+psycopg://user:pass@db:5432/neural_blocks",
        )
        self.assertEqual(
            sqlalchemy_url("postgres://user:pass@db:5432/neural_blocks"),
            "postgresql+psycopg://user:pass@db:5432/neural_blocks",
        )

    def test_migrate_bootstraps_sqlite_and_stamps_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "migration-test.db"
            environment = os.environ.copy()
            environment.pop("NBL_DATABASE_URL", None)
            environment["NBL_DATABASE_PATH"] = str(database_path)
            result = subprocess.run(
                [sys.executable, str(ROOT / "migrate.py")],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Database migrations are current.", result.stdout)
            with sqlite3.connect(database_path) as connection:
                revision = connection.execute(
                    "SELECT version_num FROM alembic_version"
                ).fetchone()
                jobs_table = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name = 'background_jobs'
                    """
                ).fetchone()
            self.assertEqual(revision, ("20260724_01",))
            self.assertEqual(jobs_table, ("background_jobs",))


if __name__ == "__main__":
    unittest.main()
