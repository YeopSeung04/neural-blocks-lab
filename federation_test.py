#!/usr/bin/env python3
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from federation import (
    AGS_LINEITEM_SCOPE,
    AGS_SCORE_SCOPE,
    FederationError,
    NRPS_MEMBERSHIP_SCOPE,
    create_ags_line_item,
    fetch_nrps_members,
    lti_authorization_url,
    oidc_authorization_url,
    post_ags_score,
    request_lti_service_token,
    validate_lti_claims,
    verify_id_token,
)

try:
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    HAS_JWT_CRYPTO = True
except ImportError:
    HAS_JWT_CRYPTO = False


PROVIDER = {
    "client_id": "client-123",
    "authorization_endpoint": "https://idp.example.edu/authorize",
    "deployment_id": "deployment-123",
}


class FederationTest(unittest.TestCase):
    def test_oidc_authorization_url_contains_state_and_nonce(self):
        url = oidc_authorization_url(
            PROVIDER,
            "state-value",
            "nonce-value",
            "https://tool.example.edu/api/auth/oidc/callback",
        )
        query = parse_qs(urlparse(url).query)
        self.assertEqual(query["response_type"], ["code"])
        self.assertEqual(query["scope"], ["openid profile email"])
        self.assertEqual(query["state"], ["state-value"])
        self.assertEqual(query["nonce"], ["nonce-value"])

    def test_lti_authorization_url_uses_form_post_id_token(self):
        url = lti_authorization_url(
            PROVIDER,
            "state-value",
            "nonce-value",
            "https://tool.example.edu/api/auth/lti/launch",
            "login-hint",
            "message-hint",
        )
        query = parse_qs(urlparse(url).query)
        self.assertEqual(query["response_type"], ["id_token"])
        self.assertEqual(query["response_mode"], ["form_post"])
        self.assertEqual(query["prompt"], ["none"])
        self.assertEqual(query["login_hint"], ["login-hint"])
        self.assertEqual(query["lti_message_hint"], ["message-hint"])

    def test_lti_claim_validation(self):
        claims = {
            "https://purl.imsglobal.org/spec/lti/claim/deployment_id": "deployment-123",
            "https://purl.imsglobal.org/spec/lti/claim/message_type": "LtiResourceLinkRequest",
            "https://purl.imsglobal.org/spec/lti/claim/version": "1.3.0",
        }
        self.assertIs(validate_lti_claims(PROVIDER, claims), claims)
        claims["https://purl.imsglobal.org/spec/lti/claim/deployment_id"] = "wrong"
        with self.assertRaises(FederationError):
            validate_lti_claims(PROVIDER, claims)

    def test_nrps_and_ags_service_requests(self):
        requests = []

        class ServiceHandler(BaseHTTPRequestHandler):
            def send_payload(self, status, payload, headers=None):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for key, value in headers or []:
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                requests.append({
                    "method": "POST",
                    "path": self.path,
                    "body": body,
                    "authorization": self.headers.get("Authorization"),
                    "contentType": self.headers.get("Content-Type"),
                })
                if self.path == "/token":
                    self.send_payload(200, {
                        "access_token": "service-token",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    })
                    return
                if self.path == "/lineitems":
                    self.send_payload(201, {
                        "id": f"http://127.0.0.1:{self.server.server_address[1]}/lineitems/1",
                        "scoreMaximum": 100,
                        "label": "XOR Lab",
                    })
                    return
                if self.path == "/lineitems/1/scores":
                    self.send_payload(200, {"status": "accepted"})
                    return
                self.send_payload(404, {"error": "not_found"})

            def do_GET(self):
                requests.append({
                    "method": "GET",
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                })
                port = self.server.server_address[1]
                if self.path == "/members?page=1":
                    self.send_payload(
                        200,
                        {"members": [{"user_id": "student-1", "status": "Active"}]},
                        [("Link", f'<http://127.0.0.1:{port}/members?page=2>; rel="next"')],
                    )
                    return
                if self.path == "/members?page=2":
                    self.send_payload(
                        200,
                        {"members": [{"user_id": "student-2", "status": "Active"}]},
                    )
                    return
                self.send_payload(404, {"error": "not_found"})

            def log_message(self, format_string, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), ServiceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        previous_secret = os.environ.get("NBL_TEST_LTI_SECRET")
        os.environ["NBL_TEST_LTI_SECRET"] = "lti-service-secret"
        try:
            port = server.server_address[1]
            provider = {
                "client_id": "client-123",
                "token_endpoint": f"http://127.0.0.1:{port}/token",
                "client_secret_env": "NBL_TEST_LTI_SECRET",
                "service_token_auth_method": "client_secret_basic",
            }
            roster = fetch_nrps_members(
                provider,
                f"http://127.0.0.1:{port}/members?page=1",
                [NRPS_MEMBERSHIP_SCOPE],
            )
            self.assertEqual(roster["pages"], 2)
            self.assertEqual(len(roster["members"]), 2)

            line_item = create_ags_line_item(
                provider,
                f"http://127.0.0.1:{port}/lineitems",
                [AGS_LINEITEM_SCOPE],
                {"id": "assignment-1", "title": "XOR Lab"},
            )
            self.assertTrue(line_item["url"].endswith("/lineitems/1"))
            score_result = post_ags_score(
                provider,
                line_item["url"],
                [AGS_SCORE_SCOPE],
                {
                    "userId": "student-1",
                    "scoreGiven": 94,
                    "scoreMaximum": 100,
                    "timestamp": "2026-07-23T00:00:00Z",
                    "comment": "Good",
                },
            )
            self.assertEqual(score_result["status"], "sent")
            token_requests = [item for item in requests if item["path"] == "/token"]
            self.assertEqual(len(token_requests), 3)
            self.assertTrue(token_requests[0]["authorization"].startswith("Basic "))
            score_requests = [
                item for item in requests if item["path"] == "/lineitems/1/scores"
            ]
            self.assertEqual(len(score_requests), 1)
            self.assertIn('"scoreGiven": 94.0', score_requests[0]["body"])
        finally:
            if previous_secret is None:
                os.environ.pop("NBL_TEST_LTI_SECRET", None)
            else:
                os.environ["NBL_TEST_LTI_SECRET"] = previous_secret
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    @unittest.skipUnless(HAS_JWT_CRYPTO, "PyJWT crypto dependencies are not installed")
    def test_private_key_jwt_service_token(self):
        captured = {}

        class TokenHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                captured.update(parse_qs(
                    self.rfile.read(length).decode("utf-8"),
                ))
                payload = json.dumps({
                    "access_token": "private-key-token",
                    "token_type": "Bearer",
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format_string, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), TokenHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")
        previous_key = os.environ.get("NBL_TEST_LTI_PRIVATE_KEY")
        os.environ["NBL_TEST_LTI_PRIVATE_KEY"] = private_pem
        try:
            endpoint = f"http://127.0.0.1:{server.server_address[1]}/token"
            token = request_lti_service_token({
                "client_id": "private-key-client",
                "token_endpoint": endpoint,
                "service_token_auth_method": "private_key_jwt",
                "private_key_env": "NBL_TEST_LTI_PRIVATE_KEY",
                "private_key_kid": "test-kid",
            }, [AGS_SCORE_SCOPE])
            self.assertEqual(token, "private-key-token")
            assertion = captured["client_assertion"][0]
            header = jwt.get_unverified_header(assertion)
            claims = jwt.decode(
                assertion,
                private_key.public_key(),
                algorithms=["RS256"],
                audience=endpoint,
            )
            self.assertEqual(header["kid"], "test-kid")
            self.assertEqual(claims["iss"], "private-key-client")
            self.assertEqual(captured["grant_type"], ["client_credentials"])
        finally:
            if previous_key is None:
                os.environ.pop("NBL_TEST_LTI_PRIVATE_KEY", None)
            else:
                os.environ["NBL_TEST_LTI_PRIVATE_KEY"] = previous_key
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    @unittest.skipUnless(HAS_JWT_CRYPTO, "PyJWT crypto dependencies are not installed")
    def test_signed_id_token_verification(self):
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
        public_jwk.update({"kid": "test-key", "use": "sig", "alg": "RS256"})
        jwks = json.dumps({"keys": [public_jwk]}).encode("utf-8")

        class JwksHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(jwks)))
                self.end_headers()
                self.wfile.write(jwks)

            def log_message(self, format_string, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), JwksHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            now = __import__("time").time()
            provider = {
                "issuer": "https://idp.example.edu",
                "client_id": "client-123",
                "jwks_uri": f"http://127.0.0.1:{server.server_address[1]}/jwks",
            }
            token = jwt.encode(
                {
                    "iss": provider["issuer"],
                    "aud": provider["client_id"],
                    "sub": "subject-123",
                    "nonce": "nonce-value",
                    "iat": int(now),
                    "exp": int(now + 300),
                },
                private_key,
                algorithm="RS256",
                headers={"kid": "test-key"},
            )
            claims = verify_id_token(provider, token, "nonce-value")
            self.assertEqual(claims["sub"], "subject-123")
            with self.assertRaises(FederationError):
                verify_id_token(provider, token, "wrong-nonce")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
