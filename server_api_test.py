import http.client
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from backend import NeuralBlocksBackend
from mailer import MemoryMailer
from server import AUTH_ATTEMPTS, AUTH_ATTEMPTS_LOCK, NeuralBlocksHandler, ThreadingHTTPServer


class ServerApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "api-test.db"
        self.previous_backend = NeuralBlocksHandler.backend
        self.backend = NeuralBlocksBackend(
            database_path,
            mailer=MemoryMailer(),
        )
        NeuralBlocksHandler.backend = self.backend
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

        status, provider_result, _ = self.request(
            "POST",
            "/api/admin/identity-providers",
            {
                "kind": "oidc",
                "name": "API University SSO",
                "issuer": "https://idp.api.test",
                "clientId": "api-client",
                "authorizationEndpoint": "https://idp.api.test/authorize",
                "tokenEndpoint": "https://idp.api.test/token",
                "jwksUri": "https://idp.api.test/jwks",
                "clientSecretEnv": "NBL_API_TEST_SECRET",
                "defaultRole": "student",
            },
            auth_headers,
        )
        self.assertEqual(status, 201)
        provider_id = provider_result["provider"]["id"]
        status, updated_provider, _ = self.request(
            "PUT",
            f"/api/admin/identity-providers/{provider_id}",
            {"enabled": False, "name": "API University SSO Disabled"},
            auth_headers,
        )
        self.assertEqual(status, 200)
        self.assertFalse(updated_provider["provider"]["enabled"])

        status, lti_service, _ = self.request(
            "GET",
            f"/api/courses/{created['course']['id']}/lti-services",
            headers={"Cookie": cookie},
        )
        self.assertEqual(status, 200)
        self.assertFalse(lti_service["service"]["connected"])

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
        self.assertEqual(self.raw_request("GET", "/.env.example"), 404)
        self.assertEqual(self.raw_request("GET", "/docker-compose.yml"), 404)
        self.assertEqual(self.raw_request("GET", "/package.json"), 404)
        self.assertEqual(self.raw_request("GET", "/catalog-test.mjs"), 404)

    def test_lti_nrps_and_ags_http_workflow(self):
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
                                "email": "lti-professor@api.test",
                                "name": "LTI Professor",
                                "status": "Active",
                                "roles": [
                                    "http://purl.imsglobal.org/vocab/lis/v2/membership#Instructor",
                                ],
                            },
                            {
                                "user_id": "student-subject",
                                "email": "lti-student@api.test",
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
                        "access_token": "lms-service-token",
                        "token_type": "Bearer",
                    })
                    return
                if self.path == "/lineitems":
                    self.send_payload(201, {
                        "id": f"http://127.0.0.1:{port}/lineitems/1",
                        "scoreMaximum": 100,
                    })
                    return
                if self.path == "/lineitems/1/scores":
                    self.send_payload(200, {"status": "accepted"})
                    return
                self.send_payload(404, {"error": "not_found"})

            def log_message(self, format_string, *args):
                return

        lms = ThreadingHTTPServer(("127.0.0.1", 0), LmsHandler)
        lms_thread = threading.Thread(target=lms.serve_forever, daemon=True)
        lms_thread.start()
        previous_secret = os.environ.get("NBL_API_LTI_SECRET")
        os.environ["NBL_API_LTI_SECRET"] = "api-lti-secret"
        try:
            registration = self.backend.register({
                "email": "lti-admin@api.test",
                "password": "lti-admin-password",
                "displayName": "LTI Admin",
                "createInstitution": True,
                "institutionName": "LTI API University",
                "institutionSlug": "lti-api-university",
            })
            self.backend.verify_email({
                "token": registration["devVerificationToken"],
            })
            admin = self.backend.authenticate(registration["sessionToken"])
            port = lms.server_address[1]
            provider_payload = self.backend.create_identity_provider(admin, {
                "kind": "lti",
                "name": "API LMS",
                "issuer": "https://lms.api.test",
                "clientId": "api-lti-client",
                "authorizationEndpoint": "https://lms.api.test/authorize",
                "tokenEndpoint": f"http://127.0.0.1:{port}/token",
                "jwksUri": "https://lms.api.test/jwks",
                "clientSecretEnv": "NBL_API_LTI_SECRET",
                "serviceTokenAuthMethod": "client_secret_basic",
                "deploymentId": "api-deployment",
                "defaultRole": "student",
            })
            provider = self.backend.get_identity_provider(
                provider_id=provider_payload["id"],
            )
            shared_claims = {
                "https://purl.imsglobal.org/spec/lti/claim/context": {
                    "id": "api-context",
                    "title": "API LTI Course",
                },
                "https://purl.imsglobal.org/spec/lti-nrps/claim/namesroleservice": {
                    "context_memberships_url": f"http://127.0.0.1:{port}/members",
                    "service_versions": ["2.0"],
                },
                "https://purl.imsglobal.org/spec/lti-ags/claim/endpoint": {
                    "scope": [
                        "https://purl.imsglobal.org/spec/lti-ags/scope/lineitem",
                        "https://purl.imsglobal.org/spec/lti-ags/scope/score",
                    ],
                    "lineitems": f"http://127.0.0.1:{port}/lineitems",
                },
            }
            professor_login = self.backend.resolve_federated_login(provider, {
                "sub": "professor-subject",
                "email": "lti-professor@api.test",
                "name": "LTI Professor",
                "https://purl.imsglobal.org/spec/lti/claim/roles": [
                    "http://purl.imsglobal.org/vocab/lis/v2/membership#Instructor",
                ],
                **shared_claims,
            })
            course_id = professor_login["courseId"]
            professor_headers = {
                "Cookie": f"nbl_session={professor_login['sessionToken']}",
                "X-CSRF-Token": professor_login["csrfToken"],
            }
            status, sync, _ = self.request(
                "POST",
                f"/api/courses/{course_id}/lti/roster-sync",
                {},
                professor_headers,
            )
            self.assertEqual(status, 200)
            self.assertEqual(sync["sync"]["received"], 2)

            student_login = self.backend.resolve_federated_login(provider, {
                "sub": "student-subject",
                "email": "lti-student@api.test",
                "name": "LTI Student",
                "https://purl.imsglobal.org/spec/lti/claim/roles": [
                    "http://purl.imsglobal.org/vocab/lis/v2/membership#Learner",
                ],
                "https://purl.imsglobal.org/spec/lti/claim/context": {
                    "id": "api-context",
                },
            })
            professor = self.backend.authenticate(professor_login["sessionToken"])
            student = self.backend.authenticate(student_login["sessionToken"])
            assignment = self.backend.create_assignment(professor, course_id, {
                "title": "API LTI Assignment",
                "requiredFamily": "mlp",
            })
            snapshot = {
                "model": {"family": "mlp"},
                "result": {"validationAccuracy": 0.9, "validationLoss": 0.2},
            }
            project = self.backend.save_project(student, course_id, {
                "name": "API LTI Project",
                "snapshot": snapshot,
            })
            submission = self.backend.submit_assignment(student, assignment["id"], {
                "projectId": project["id"],
                "snapshot": snapshot,
            })
            self.backend.grade_submission(professor, submission["id"], {
                "score": 92,
                "feedback": "Ready for LMS",
            })
            status, passback, _ = self.request(
                "POST",
                f"/api/submissions/{submission['id']}/lti-grade-passback",
                {},
                professor_headers,
            )
            self.assertEqual(status, 200)
            self.assertEqual(passback["passback"]["status"], "sent")
            self.assertIn(("GET", "/members"), external_requests)
            self.assertTrue(any(
                request[0:2] == ("POST", "/lineitems/1/scores")
                for request in external_requests
            ))
        finally:
            if previous_secret is None:
                os.environ.pop("NBL_API_LTI_SECRET", None)
            else:
                os.environ["NBL_API_LTI_SECRET"] = previous_secret
            lms.shutdown()
            lms.server_close()
            lms_thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
