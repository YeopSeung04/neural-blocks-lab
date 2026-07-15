#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from backend import ApiError, NeuralBlocksBackend


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.backend = NeuralBlocksBackend(Path(self.temp.name) / "test.db")

    def tearDown(self):
        self.temp.cleanup()

    def register_institution(self, suffix):
        return self.backend.register({
            "email": f"professor-{suffix}@example.edu",
            "password": "correct-horse-battery-staple",
            "displayName": f"Professor {suffix}",
            "createInstitution": True,
            "institutionName": f"University {suffix}",
            "institutionSlug": f"university-{suffix}",
        })

    def test_auth_tenant_isolation_and_classroom_flow(self):
        tenant_a = self.register_institution("alpha")
        tenant_b = self.register_institution("beta")
        auth_a = self.backend.authenticate(tenant_a["sessionToken"])
        auth_b = self.backend.authenticate(tenant_b["sessionToken"])
        self.assertNotEqual(auth_a["tenant"]["id"], auth_b["tenant"]["id"])

        login_a = self.backend.login({
            "email": "professor-alpha@example.edu",
            "password": "correct-horse-battery-staple",
        })
        self.assertEqual(login_a["user"]["tenantId"], auth_a["tenant"]["id"])

        course_a = self.backend.create_course(auth_a, {
            "name": "AI Foundations",
            "code": "AI101",
            "term": "2026-2",
        })
        course_b = self.backend.create_course(auth_b, {
            "name": "Machine Learning",
            "code": "ML201",
            "term": "2026-2",
        })
        self.assertEqual(len(self.backend.list_courses(auth_a)), 1)
        self.assertEqual(len(self.backend.list_courses(auth_b)), 1)
        self.assertNotEqual(course_a["id"], course_b["id"])

        student_registration = self.backend.register({
            "email": "student-alpha@example.edu",
            "password": "student-password-1234",
            "displayName": "Student Alpha",
            "createInstitution": False,
            "institutionJoinCode": auth_a["tenant"]["joinCode"],
        })
        student_auth = self.backend.authenticate(student_registration["sessionToken"])
        joined_course = self.backend.join_course(
            student_auth,
            {"joinCode": course_a["joinCode"]},
        )
        self.assertIsNone(joined_course["joinCode"])
        self.assertIsNone(self.backend.list_courses(student_auth)[0]["joinCode"])
        self.assertEqual(len(self.backend.list_courses(student_auth)), 1)

        with self.assertRaises(ApiError) as tenant_error:
            self.backend.list_assignments(student_auth, course_b["id"])
        self.assertEqual(tenant_error.exception.status, 404)

        assignment = self.backend.create_assignment(auth_a, course_a["id"], {
            "title": "XOR Lab",
            "instructions": "Reach validation accuracy 80%.",
            "requiredFamily": "mlp",
            "targetAccuracy": 0.8,
            "starterSnapshot": {
                "model": {"family": "mlp"},
                "result": {"validationAccuracy": 0.5},
            },
        })
        self.assertEqual(len(self.backend.list_assignments(student_auth, course_a["id"])), 1)

        snapshot = {
            "model": {"family": "mlp", "parameterCount": 33},
            "result": {"validationAccuracy": 0.91, "validationLoss": 0.2},
        }
        project = self.backend.save_project(student_auth, course_a["id"], {
            "name": "Student XOR",
            "snapshot": snapshot,
        })
        self.assertEqual(project["versionCount"], 1)
        loaded_project = self.backend.get_project(student_auth, project["id"])
        self.assertEqual(
            loaded_project["latestSnapshot"]["result"]["validationAccuracy"],
            0.91,
        )

        submission = self.backend.submit_assignment(student_auth, assignment["id"], {
            "projectId": project["id"],
            "snapshot": snapshot,
        })
        self.assertTrue(submission["autoEvaluation"]["passed"])
        self.assertEqual(submission["attempt"], 1)

        graded = self.backend.grade_submission(auth_a, submission["id"], {
            "score": 94,
            "feedback": "Reproducible experiment.",
        })
        self.assertEqual(graded["status"], "graded")
        self.assertEqual(graded["score"], 94)
        self.assertEqual(
            self.backend.list_submissions(student_auth, course_a["id"])[0]["score"],
            94,
        )
        self.assertEqual(len(self.backend.list_submissions(auth_b, course_b["id"])), 0)


if __name__ == "__main__":
    unittest.main()
