#!/usr/bin/env python3
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from asgi_app import create_app


class AsgiAppTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database_path = self.root / "asgi.db"
        (self.root / "index.html").write_text(
            "<!doctype html><title>ASGI Test</title>",
            encoding="utf-8",
        )
        (self.root / "app.js").write_text("export {};", encoding="utf-8")
        self.app = create_app(
            database_target=self.database_path,
            base_url="http://testserver",
            root=self.root,
            job_mode="inline",
            payload_key=Fernet.generate_key().decode("ascii"),
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def register_admin(self):
        response = self.client.post("/api/auth/register", json={
            "createInstitution": True,
            "institutionName": "ASGI University",
            "institutionSlug": "asgi-university",
            "displayName": "ASGI Admin",
            "email": "admin@asgi.test",
            "password": "asgi-admin-password",
        })
        self.assertEqual(response.status_code, 201)
        registration = response.json()
        verified = self.client.post("/api/auth/verify-email", json={
            "token": registration["devVerificationToken"],
        })
        self.assertEqual(verified.status_code, 200)
        return registration

    def test_fastapi_auth_static_security_and_health(self):
        registration = self.register_admin()
        response = self.client.post(
            "/api/courses",
            headers={"X-CSRF-Token": registration["csrfToken"]},
            json={"name": "AI", "code": "AI101", "term": "2026-2"},
        )
        self.assertEqual(response.status_code, 201)
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["server"], "fastapi")
        self.assertEqual(health.json()["queue"]["mode"], "inline")
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/.env.example").status_code, 404)
        self.assertEqual(self.client.get("/backend.py").status_code, 404)
        self.assertEqual(self.client.get("/alembic.ini").status_code, 404)
        self.assertEqual(
            self.client.get("/api/jobs").status_code,
            200,
        )

    def test_lti_jobs_run_through_inline_worker(self):
        external_requests = []

        class LmsHandler(BaseHTTPRequestHandler):
            def send_payload(self, status, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                external_requests.append(("GET", self.path))
                if self.path == "/members":
                    self.send_payload(200, {
                        "members": [
                            {
                                "user_id": "professor-subject",
                                "email": "professor@lti.asgi",
                                "name": "LTI Professor",
                                "status": "Active",
                                "roles": [
                                    "http://purl.imsglobal.org/vocab/lis/v2/membership#Instructor",
                                ],
                            },
                            {
                                "user_id": "student-subject",
                                "email": "student@lti.asgi",
                                "name": "LTI Student",
                                "status": "Active",
                                "roles": [
                                    "http://purl.imsglobal.org/vocab/lis/v2/membership#Learner",
                                ],
                            },
                        ],
                    })
                    return
                self.send_payload(404, {"error": "not_found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                external_requests.append(("POST", self.path, body))
                port = self.server.server_address[1]
                if self.path == "/token":
                    self.send_payload(200, {
                        "access_token": "asgi-lti-token",
                        "token_type": "Bearer",
                    })
                    return
                if self.path == "/lineitems":
                    self.send_payload(201, {
                        "id": f"http://127.0.0.1:{port}/lineitems/1",
                    })
                    return
                if self.path == "/lineitems/1/scores":
                    self.send_payload(200, {"status": "accepted"})
                    return
                self.send_payload(404, {"error": "not_found"})

            def log_message(self, format_string, *args):
                return

        lms = ThreadingHTTPServer(("127.0.0.1", 0), LmsHandler)
        thread = threading.Thread(target=lms.serve_forever, daemon=True)
        thread.start()
        previous_secret = os.environ.get("NBL_ASGI_LTI_SECRET")
        os.environ["NBL_ASGI_LTI_SECRET"] = "asgi-lti-secret"
        try:
            registration = self.register_admin()
            backend = self.app.state.backend
            admin = backend.authenticate(
                self.client.cookies.get("nbl_session"),
            )
            port = lms.server_address[1]
            provider_payload = backend.create_identity_provider(admin, {
                "kind": "lti",
                "name": "ASGI LMS",
                "issuer": "https://lms.asgi.test",
                "clientId": "asgi-lti-client",
                "authorizationEndpoint": "https://lms.asgi.test/authorize",
                "tokenEndpoint": f"http://127.0.0.1:{port}/token",
                "jwksUri": "https://lms.asgi.test/jwks",
                "clientSecretEnv": "NBL_ASGI_LTI_SECRET",
                "deploymentId": "asgi-deployment",
                "defaultRole": "student",
            })
            provider = backend.get_identity_provider(
                provider_id=provider_payload["id"],
            )
            claims = {
                "https://purl.imsglobal.org/spec/lti/claim/context": {
                    "id": "asgi-context",
                    "title": "ASGI LTI Course",
                },
                "https://purl.imsglobal.org/spec/lti-nrps/claim/namesroleservice": {
                    "context_memberships_url": f"http://127.0.0.1:{port}/members",
                },
                "https://purl.imsglobal.org/spec/lti-ags/claim/endpoint": {
                    "scope": [
                        "https://purl.imsglobal.org/spec/lti-ags/scope/lineitem",
                        "https://purl.imsglobal.org/spec/lti-ags/scope/score",
                    ],
                    "lineitems": f"http://127.0.0.1:{port}/lineitems",
                },
            }
            professor_login = backend.resolve_federated_login(provider, {
                "sub": "professor-subject",
                "email": "professor@lti.asgi",
                "name": "LTI Professor",
                "https://purl.imsglobal.org/spec/lti/claim/roles": [
                    "http://purl.imsglobal.org/vocab/lis/v2/membership#Instructor",
                ],
                **claims,
            })
            course_id = professor_login["courseId"]
            sync = self.client.post(
                f"/api/courses/{course_id}/lti/roster-sync",
                headers={"X-CSRF-Token": registration["csrfToken"]},
                json={},
            )
            self.assertEqual(sync.status_code, 202)
            self.assertEqual(sync.json()["job"]["status"], "succeeded")
            self.assertEqual(sync.json()["job"]["result"]["received"], 2)

            student_login = backend.resolve_federated_login(provider, {
                "sub": "student-subject",
                "email": "student@lti.asgi",
                "name": "LTI Student",
                "https://purl.imsglobal.org/spec/lti/claim/roles": [
                    "http://purl.imsglobal.org/vocab/lis/v2/membership#Learner",
                ],
                "https://purl.imsglobal.org/spec/lti/claim/context": {
                    "id": "asgi-context",
                },
            })
            professor = backend.authenticate(professor_login["sessionToken"])
            student = backend.authenticate(student_login["sessionToken"])
            assignment = backend.create_assignment(professor, course_id, {
                "title": "ASGI Job Assignment",
                "requiredFamily": "mlp",
            })
            snapshot = {
                "model": {"family": "mlp"},
                "result": {"validationAccuracy": 0.9, "validationLoss": 0.2},
            }
            project = backend.save_project(student, course_id, {
                "name": "ASGI Project",
                "snapshot": snapshot,
            })
            submission = backend.submit_assignment(student, assignment["id"], {
                "projectId": project["id"],
                "snapshot": snapshot,
            })
            backend.grade_submission(professor, submission["id"], {
                "score": 93,
                "feedback": "Queue verified",
            })
            passback = self.client.post(
                f"/api/submissions/{submission['id']}/lti-grade-passback",
                headers={"X-CSRF-Token": registration["csrfToken"]},
                json={},
            )
            self.assertEqual(passback.status_code, 202)
            self.assertEqual(passback.json()["job"]["status"], "succeeded")
            self.assertEqual(passback.json()["job"]["result"]["score"], 93)
            jobs = self.client.get("/api/jobs").json()["jobs"]
            self.assertEqual(len(jobs), 2)
            self.assertTrue(any(
                item[0:2] == ("POST", "/lineitems/1/scores")
                for item in external_requests
            ))
        finally:
            if previous_secret is None:
                os.environ.pop("NBL_ASGI_LTI_SECRET", None)
            else:
                os.environ["NBL_ASGI_LTI_SECRET"] = previous_secret
            lms.shutdown()
            lms.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
