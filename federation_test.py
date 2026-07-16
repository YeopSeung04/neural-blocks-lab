#!/usr/bin/env python3
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from federation import (
    FederationError,
    lti_authorization_url,
    oidc_authorization_url,
    validate_lti_claims,
    verify_id_token,
)

try:
    import jwt
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
