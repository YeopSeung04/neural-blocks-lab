#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from cryptography.fernet import Fernet

from job_queue import JobExecutionError, JobManager


class JobQueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp.name) / "jobs.db"
        self.payload_key = Fernet.generate_key().decode("ascii")

    def tearDown(self):
        self.temp.cleanup()

    def test_inline_job_success_and_sensitive_payload_cleanup(self):
        received = []

        def execute(job_type, payload):
            received.append((job_type, payload))
            return {"value": payload["value"] * 2}

        jobs = JobManager(
            self.database_path,
            mode="inline",
            payload_key=self.payload_key,
            executor=execute,
        )
        job = jobs.enqueue(
            "test.double",
            {"value": 21, "secret": "not-stored-after-success"},
            tenant_id="tenant_1",
            user_id="user_1",
            sensitive=True,
        )
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["result"]["value"], 42)
        self.assertEqual(received[0][0], "test.double")
        with jobs.database.connect() as db:
            row = db.execute(
                "SELECT payload_text FROM background_jobs WHERE id = ?",
                (job["id"],),
            ).fetchone()
        self.assertEqual(row["payload_text"], "")
        listed = jobs.list_for_auth({
            "user": {"tenantId": "tenant_1"},
        })
        self.assertEqual(listed[0]["id"], job["id"])

    def test_inline_job_retries_then_fails(self):
        attempts = []

        def fail(job_type, payload):
            attempts.append((job_type, payload))
            raise JobExecutionError("Temporary failure", "temporary_failure")

        jobs = JobManager(
            self.database_path,
            mode="inline",
            payload_key=self.payload_key,
            executor=fail,
        )
        job = jobs.enqueue(
            "test.fail",
            {"value": 1},
            tenant_id="tenant_1",
            user_id="user_1",
            max_attempts=3,
        )
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["attempt"], 3)
        self.assertEqual(job["error"]["code"], "temporary_failure")
        self.assertEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()
