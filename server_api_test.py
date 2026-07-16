import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from backend import NeuralBlocksBackend
from mailer import MemoryMailer
from server import AUTH_ATTEMPTS, AUTH_ATTEMPTS_LOCK, NeuralBlocksHandler, ThreadingHTTPServer


class ServerApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "api-test.db"
        self.previous_backend = NeuralBlocksHandler.backend
        NeuralBlocksHandler.backend = NeuralBlocksBackend(
            database_path,
            mailer=MemoryMailer(),
        )
        with AUTH_ATTEMPTS_LOCK:
            AUTH_ATTEMPTS.clear()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), NeuralBlocksHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        NeuralBlocksHandler.backend = self.previous_backend
        self.temporary_directory.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        request_headers = dict(headers or {})
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        connection.request(method, path, body=payload, headers=request_headers)
        response = connection.getresponse()
        raw_body = response.read()
        response_headers = dict(response.getheaders())
        connection.close()
        return response.status, json.loads(raw_body), response_headers

    def raw_request(self, method, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(method, path)
        response = connection.getresponse()
        response.read()
        status = response.status
        connection.close()
        return status

    def test_session_cookie_and_csrf_protection(self):
        status, registration, headers = self.request(
            "POST",
            "/api/auth/register",
            {
                "createInstitution": True,
                "institutionName": "API Test University",
                "institutionSlug": "api-test-university",
                "displayName": "API Admin",
                "email": "admin@api.test",
                "password": "CorrectHorseBatteryStaple",
            },
        )
        self.assertEqual(status, 201)
        cookie_header = headers["Set-Cookie"]
        self.assertIn("HttpOnly", cookie_header)
        self.assertIn("SameSite=Lax", cookie_header)
        cookie = cookie_header.split(";", 1)[0]

        status, error, _ = self.request(
            "POST",
            "/api/courses",
            {"name": "AI 101", "code": "AI101", "term": "2026-2"},
            {"Cookie": cookie},
        )
        self.assertEqual(status, 403)
        self.assertEqual(error["error"]["code"], "csrf_failed")

        auth_headers = {
            "Cookie": cookie,
            "X-CSRF-Token": registration["csrfToken"],
        }
        status, unverified, _ = self.request(
            "POST",
            "/api/courses",
            {"name": "AI 101", "code": "AI101", "term": "2026-2"},
            auth_headers,
        )
        self.assertEqual(status, 403)
        self.assertEqual(unverified["error"]["code"], "email_not_verified")

        status, verified, _ = self.request(
            "POST",
            "/api/auth/verify-email",
            {"token": registration["devVerificationToken"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(verified["status"], "verified")

        status, created, _ = self.request(
            "POST",
            "/api/courses",
            {"name": "AI 101", "code": "AI101", "term": "2026-2"},
            auth_headers,
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["course"]["code"], "AI101")

        status, current_user, _ = self.request(
            "GET",
            "/api/auth/me",
            headers={"Cookie": cookie},
        )
        self.assertEqual(status, 200)
        self.assertEqual(current_user["tenant"]["slug"], "api-test-university")
        self.assertTrue(current_user["user"]["emailVerified"])

        status, invitation, _ = self.request(
            "POST",
            "/api/admin/invitations",
            {
                "email": "professor-invite@api.test",
                "role": "professor",
                "courseId": created["course"]["id"],
            },
            auth_headers,
        )
        self.assertEqual(status, 201)
        status, accepted, accepted_headers = self.request(
            "POST",
            "/api/auth/invitations/accept",
            {
                "token": invitation["invitation"]["devInvitationToken"],
                "displayName": "Professor Invite",
                "password": "ProfessorInvitePassword",
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(accepted["user"]["role"], "professor")
        self.assertIn("HttpOnly", accepted_headers["Set-Cookie"])

        status, roster, _ = self.request(
            "GET",
            f"/api/courses/{created['course']['id']}/members",
            headers={"Cookie": cookie},
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(roster["members"]), 2)

        status, audit, _ = self.request(
            "GET",
            "/api/admin/audit?limit=20",
            headers={"Cookie": cookie},
        )
        self.assertEqual(status, 200)
        self.assertTrue(audit["events"])

        status, logged_out, logout_headers = self.request(
            "POST",
            "/api/auth/logout",
            {},
            auth_headers,
        )
        self.assertEqual(status, 200)
        self.assertEqual(logged_out["status"], "logged_out")
        self.assertIn("Max-Age=0", logout_headers["Set-Cookie"])

        status, unauthorized, _ = self.request(
            "GET",
            "/api/auth/me",
            headers={"Cookie": cookie},
        )
        self.assertEqual(status, 401)
        self.assertEqual(unauthorized["error"]["code"], "unauthorized")

    def test_backend_sources_are_not_public_static_files(self):
        self.assertEqual(self.raw_request("GET", "/backend.py"), 404)
        self.assertEqual(self.raw_request("HEAD", "/server.py"), 404)
        self.assertEqual(self.raw_request("GET", "/.data/neural_blocks.db"), 404)


if __name__ == "__main__":
    unittest.main()
