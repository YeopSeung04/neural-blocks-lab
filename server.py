#!/usr/bin/env python3
import argparse
import hmac
import json
import os
import re
import threading
import time
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from backend import ApiError, NeuralBlocksBackend, iso_time, parse_json
from federation import (
    FederationError,
    create_ags_line_item,
    exchange_oidc_code,
    fetch_nrps_members,
    lti_authorization_url,
    oidc_authorization_url,
    post_ags_score,
    validate_lti_claims,
    verify_id_token,
)
from mailer import Mailer
from system_metrics import collect_metrics

ROOT = Path(__file__).resolve().parent
DATABASE_TARGET = os.environ.get(
    "NBL_DATABASE_URL",
    os.environ.get("NBL_DATABASE_PATH", ROOT / ".data" / "neural_blocks.db"),
)
BASE_URL = os.environ.get("NBL_BASE_URL", "http://127.0.0.1:8770").rstrip("/")
EXPOSE_DEV_TOKENS = os.environ.get("NBL_EXPOSE_DEV_TOKENS", "1") == "1"
AUTH_WINDOW_SECONDS = 15 * 60
AUTH_MAX_ATTEMPTS = 12
AUTH_ATTEMPTS = {}
AUTH_ATTEMPTS_LOCK = threading.Lock()


BACKEND = NeuralBlocksBackend(
    DATABASE_TARGET,
    mailer=Mailer(ROOT / ".data" / "mail-outbox.jsonl"),
    base_url=BASE_URL,
    expose_dev_tokens=EXPOSE_DEV_TOKENS,
)


def allow_auth_attempt(key):
    now = time.time()
    with AUTH_ATTEMPTS_LOCK:
        recent = [
            attempt for attempt in AUTH_ATTEMPTS.get(key, [])
            if now - attempt < AUTH_WINDOW_SECONDS
        ]
        if len(recent) >= AUTH_MAX_ATTEMPTS:
            AUTH_ATTEMPTS[key] = recent
            return False
        recent.append(now)
        AUTH_ATTEMPTS[key] = recent
        return True


class NeuralBlocksHandler(SimpleHTTPRequestHandler):
    backend = BACKEND
    max_json_bytes = 2 * 1024 * 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=()",
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "worker-src 'self' blob:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'self'",
        )
        super().end_headers()

    def api_path(self):
        return urlparse(self.path).path

    def protected_static_path(self, path):
        decoded = unquote(path).replace("\\", "/")
        parts = [part for part in decoded.split("/") if part not in ("", ".")]
        if any(
            part.startswith(".") or part in {"__pycache__"}
            for part in parts
        ):
            return True
        filename = parts[-1].lower() if parts else ""
        if filename in {
            "dockerfile",
            "docker-compose.yml",
            "package.json",
            "package-lock.json",
            "requirements.txt",
            "readme.md",
            "alembic.ini",
        }:
            return True
        return (
            filename.endswith((
                ".db",
                ".db-shm",
                ".db-wal",
                ".py",
                ".pyc",
                ".mako",
                "-test.mjs",
                "_test.mjs",
            ))
            or filename == "test.mjs"
        )

    def send_json(self, status, data, headers=None):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        for key, value in headers or []:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def read_json(self):
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ApiError(415, "Content-Type must be application/json", "unsupported_media_type")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError(400, "Invalid Content-Length", "invalid_request")
        if length <= 0:
            return {}
        if length > self.max_json_bytes:
            raise ApiError(413, "JSON request body is too large", "payload_too_large")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError(400, "Request body must be valid JSON", "invalid_json")
        if not isinstance(payload, dict):
            raise ApiError(400, "JSON request body must be an object", "invalid_json")
        return payload

    def read_form(self):
        content_type = self.headers.get("Content-Type", "")
        if "application/x-www-form-urlencoded" not in content_type:
            raise ApiError(
                415,
                "Content-Type must be application/x-www-form-urlencoded",
                "unsupported_media_type",
            )
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError(400, "Invalid Content-Length", "invalid_request")
        if length <= 0 or length > self.max_json_bytes:
            raise ApiError(400, "Form request body is invalid", "invalid_request")
        try:
            values = parse_qs(
                self.rfile.read(length).decode("utf-8"),
                keep_blank_values=True,
                max_num_fields=50,
            )
        except (UnicodeDecodeError, ValueError):
            raise ApiError(400, "Form request body is invalid", "invalid_request")
        return {key: items[-1] for key, items in values.items()}

    def session_token(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return None
        value = cookie.get("nbl_session")
        return value.value if value else None

    def require_auth(self):
        auth = self.backend.authenticate(self.session_token())
        auth["_request"] = self.request_context()
        return auth

    def request_context(self):
        return {
            "ip": self.client_address[0],
            "userAgent": self.headers.get("User-Agent", ""),
        }

    def public_auth(self, auth):
        return {
            key: value
            for key, value in auth.items()
            if not str(key).startswith("_")
        }

    def require_csrf(self, auth):
        provided = self.headers.get("X-CSRF-Token", "")
        expected = auth.get("csrfToken", "")
        if not provided or not hmac.compare_digest(provided, expected):
            raise ApiError(403, "CSRF token is invalid", "csrf_failed")

    def session_cookie(self, token, clear=False):
        parts = [
            f"nbl_session={'' if clear else token}",
            "Path=/",
            "HttpOnly",
            "SameSite=Lax",
            f"Max-Age={0 if clear else 7 * 24 * 60 * 60}",
        ]
        if os.environ.get("NBL_SECURE_COOKIES") == "1":
            parts.append("Secure")
        return "; ".join(parts)

    def auth_rate_key(self, payload):
        email = str(payload.get("email") or "").strip().lower()
        return f"{self.client_address[0]}:{email}"

    def handle_api_error(self, error):
        self.send_json(
            error.status,
            {"error": {"code": error.code, "message": error.message}},
        )

    def send_redirect(self, location, headers=None, status=302):
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        for key, value in headers or []:
            self.send_header(key, value)
        self.end_headers()

    def safe_return_path(self, value):
        value = str(value or "/").strip()
        parsed = urlparse(value)
        if parsed.scheme or parsed.netloc:
            base = urlparse(BASE_URL)
            if parsed.scheme != base.scheme or parsed.netloc != base.netloc:
                return "/"
            value = parsed.path or "/"
            if parsed.query:
                value += f"?{parsed.query}"
        if not value.startswith("/") or value.startswith("//"):
            return "/"
        return value[:1000]

    def handle_federation_error(self, error):
        self.send_json(
            error.status,
            {"error": {"code": error.code, "message": error.message}},
        )

    def do_GET(self):
        path = self.api_path()
        query = parse_qs(urlparse(self.path).query)
        if path == "/api/system-metrics":
            self.send_json(200, collect_metrics())
            return
        if path == "/api/health":
            self.send_json(
                200,
                {"status": "ok", "database": self.backend.database.description},
            )
            return
        if path == "/api/auth/providers":
            try:
                tenant_slug = (query.get("tenant") or [""])[-1]
                self.send_json(
                    200,
                    {"providers": self.backend.public_identity_providers(tenant_slug)},
                )
            except ApiError as error:
                self.handle_api_error(error)
            return
        if path == "/api/auth/oidc/start":
            try:
                tenant_slug = (query.get("tenant") or [""])[-1]
                provider_id = (query.get("provider") or [""])[-1]
                return_to = self.safe_return_path((query.get("returnTo") or ["/"])[-1])
                provider = self.backend.get_identity_provider(
                    provider_id=provider_id,
                    tenant_slug=tenant_slug,
                    kind="oidc",
                )
                state = self.backend.create_federation_state(
                    provider,
                    "oidc",
                    return_to,
                )
                redirect_uri = f"{BASE_URL}/api/auth/oidc/callback"
                self.send_redirect(
                    oidc_authorization_url(
                        provider,
                        state["state"],
                        state["nonce"],
                        redirect_uri,
                    )
                )
            except ApiError as error:
                self.handle_api_error(error)
            except FederationError as error:
                self.handle_federation_error(error)
            return
        if path == "/api/auth/oidc/callback":
            try:
                code = (query.get("code") or [""])[-1]
                state_value = (query.get("state") or [""])[-1]
                if not code:
                    raise ApiError(400, "OIDC authorization code is missing", "invalid_request")
                state = self.backend.consume_federation_state(state_value, "oidc")
                provider = state["provider"]
                redirect_uri = f"{BASE_URL}/api/auth/oidc/callback"
                token_response = exchange_oidc_code(provider, code, redirect_uri)
                claims = verify_id_token(
                    provider,
                    token_response["id_token"],
                    state["nonce"],
                )
                auth = self.backend.resolve_federated_login(
                    provider,
                    claims,
                    self.request_context(),
                )
                session_token = auth.pop("sessionToken")
                self.send_redirect(
                    self.safe_return_path(state["targetPath"]),
                    [("Set-Cookie", self.session_cookie(session_token))],
                )
            except ApiError as error:
                self.handle_api_error(error)
            except FederationError as error:
                self.handle_federation_error(error)
            return
        if path.startswith("/api/"):
            try:
                auth = self.require_auth()
                if path == "/api/auth/me":
                    self.send_json(200, self.public_auth(auth))
                    return
                if path == "/api/admin/invitations":
                    self.send_json(
                        200,
                        {"invitations": self.backend.list_invitations(auth)},
                    )
                    return
                if path == "/api/admin/audit":
                    limit = (query.get("limit") or ["100"])[-1]
                    self.send_json(
                        200,
                        {"events": self.backend.list_audit_events(auth, limit)},
                    )
                    return
                if path == "/api/admin/identity-providers":
                    self.send_json(
                        200,
                        {"providers": self.backend.list_identity_providers(auth)},
                    )
                    return
                if path == "/api/courses":
                    self.send_json(200, {"courses": self.backend.list_courses(auth)})
                    return
                match = re.fullmatch(r"/api/courses/([^/]+)/assignments", path)
                if match:
                    self.send_json(
                        200,
                        {"assignments": self.backend.list_assignments(auth, match.group(1))},
                    )
                    return
                match = re.fullmatch(r"/api/courses/([^/]+)/projects", path)
                if match:
                    self.send_json(
                        200,
                        {"projects": self.backend.list_projects(auth, match.group(1))},
                    )
                    return
                match = re.fullmatch(r"/api/courses/([^/]+)/submissions", path)
                if match:
                    self.send_json(
                        200,
                        {"submissions": self.backend.list_submissions(auth, match.group(1))},
                    )
                    return
                match = re.fullmatch(r"/api/courses/([^/]+)/members", path)
                if match:
                    self.send_json(
                        200,
                        {"members": self.backend.list_course_members(auth, match.group(1))},
                    )
                    return
                match = re.fullmatch(r"/api/courses/([^/]+)/lti-services", path)
                if match:
                    self.send_json(
                        200,
                        {
                            "service": self.backend.get_lti_course_service(
                                auth,
                                match.group(1),
                            )
                        },
                    )
                    return
                match = re.fullmatch(r"/api/projects/([^/]+)", path)
                if match:
                    self.send_json(200, {"project": self.backend.get_project(auth, match.group(1))})
                    return
                raise ApiError(404, "API endpoint was not found", "not_found")
            except ApiError as error:
                self.handle_api_error(error)
            except Exception as error:
                self.log_error("Unhandled API GET error: %s", error)
                self.send_json(500, {"error": {"code": "server_error", "message": "Internal server error"}})
            return
        if self.protected_static_path(path):
            self.send_error(404, "Not Found")
            return
        super().do_GET()

    def do_HEAD(self):
        path = self.api_path()
        if path.startswith("/api/") or self.protected_static_path(path):
            self.send_error(404, "Not Found")
            return
        super().do_HEAD()

    def do_POST(self):
        path = self.api_path()
        if not path.startswith("/api/"):
            self.send_error(405, "Method Not Allowed")
            return
        try:
            if path in ("/api/auth/lti/login", "/api/auth/lti/launch"):
                form = self.read_form()
                if path.endswith("/login"):
                    issuer = str(form.get("iss") or "").rstrip("/")
                    client_id = form.get("client_id")
                    provider = self.backend.get_identity_provider(
                        issuer=issuer,
                        client_id=client_id,
                        kind="lti",
                    )
                    deployment_id = form.get("lti_deployment_id")
                    if deployment_id and deployment_id != provider["deployment_id"]:
                        raise ApiError(
                            400,
                            "LTI deployment ID is invalid",
                            "invalid_lti_deployment",
                        )
                    login_hint = str(form.get("login_hint") or "")
                    if not login_hint:
                        raise ApiError(400, "LTI login_hint is required", "invalid_request")
                    target_path = self.safe_return_path(
                        form.get("target_link_uri") or "/"
                    )
                    state = self.backend.create_federation_state(
                        provider,
                        "lti",
                        target_path,
                        {
                            "ltiMessageHint": form.get("lti_message_hint"),
                            "deploymentId": deployment_id,
                        },
                    )
                    self.send_redirect(
                        lti_authorization_url(
                            provider,
                            state["state"],
                            state["nonce"],
                            f"{BASE_URL}/api/auth/lti/launch",
                            login_hint,
                            form.get("lti_message_hint"),
                        )
                    )
                    return

                state = self.backend.consume_federation_state(
                    form.get("state"),
                    "lti",
                )
                provider = state["provider"]
                claims = verify_id_token(
                    provider,
                    form.get("id_token"),
                    state["nonce"],
                )
                validate_lti_claims(provider, claims)
                auth = self.backend.resolve_federated_login(
                    provider,
                    claims,
                    self.request_context(),
                )
                session_token = auth.pop("sessionToken")
                destination = self.safe_return_path(state["targetPath"])
                separator = "&" if "?" in destination else "?"
                self.send_redirect(
                    f"{destination}{separator}lti=launched",
                    [("Set-Cookie", self.session_cookie(session_token))],
                )
                return

            payload = self.read_json()
            public_auth_paths = {
                "/api/auth/register",
                "/api/auth/login",
                "/api/auth/invitations/accept",
                "/api/auth/verify-email",
                "/api/auth/resend-verification",
                "/api/auth/password-reset/request",
                "/api/auth/password-reset/confirm",
            }
            if path in public_auth_paths:
                key = self.auth_rate_key(payload)
                if not allow_auth_attempt(key):
                    raise ApiError(429, "Too many authentication attempts", "rate_limited")
                if path == "/api/auth/register":
                    auth = self.backend.register(payload, self.request_context())
                    token = auth.pop("sessionToken")
                    self.send_json(
                        201,
                        auth,
                        [("Set-Cookie", self.session_cookie(token))],
                    )
                    return
                if path == "/api/auth/login":
                    auth = self.backend.login(payload, self.request_context())
                    token = auth.pop("sessionToken")
                    self.send_json(
                        200,
                        auth,
                        [("Set-Cookie", self.session_cookie(token))],
                    )
                    return
                if path == "/api/auth/invitations/accept":
                    auth = self.backend.accept_invitation(
                        payload,
                        self.request_context(),
                    )
                    token = auth.pop("sessionToken")
                    self.send_json(
                        201,
                        auth,
                        [("Set-Cookie", self.session_cookie(token))],
                    )
                    return
                if path == "/api/auth/verify-email":
                    self.send_json(
                        200,
                        self.backend.verify_email(payload, self.request_context()),
                    )
                    return
                if path == "/api/auth/resend-verification":
                    self.send_json(202, self.backend.resend_verification(payload))
                    return
                if path == "/api/auth/password-reset/request":
                    self.send_json(202, self.backend.request_password_reset(payload))
                    return
                if path == "/api/auth/password-reset/confirm":
                    self.send_json(
                        200,
                        self.backend.confirm_password_reset(
                            payload,
                            self.request_context(),
                        ),
                    )
                    return
                return

            auth = self.require_auth()
            self.require_csrf(auth)
            if path == "/api/auth/logout":
                self.backend.logout(self.session_token())
                self.send_json(
                    200,
                    {"status": "logged_out"},
                    [("Set-Cookie", self.session_cookie("", clear=True))],
                )
                return
            if path == "/api/courses":
                self.send_json(201, {"course": self.backend.create_course(auth, payload)})
                return
            if path == "/api/admin/invitations":
                self.send_json(
                    201,
                    {"invitation": self.backend.create_invitation(auth, payload)},
                )
                return
            if path == "/api/admin/identity-providers":
                self.send_json(
                    201,
                    {"provider": self.backend.create_identity_provider(auth, payload)},
                )
                return
            match = re.fullmatch(r"/api/courses/([^/]+)/lti/roster-sync", path)
            if match:
                course_id = match.group(1)
                service = self.backend.get_lti_course_service(
                    auth,
                    course_id,
                    include_private=True,
                )
                if not service["connected"] or not service["nrps"]["available"]:
                    raise ApiError(
                        409,
                        "NRPS membership service is unavailable",
                        "nrps_unavailable",
                    )
                if not service["provider"]["enabled"]:
                    raise ApiError(
                        409,
                        "LTI provider is disabled",
                        "lti_provider_disabled",
                    )
                provider = service.pop("_provider")
                roster = fetch_nrps_members(
                    provider,
                    service["nrps"]["membershipsUrl"],
                    service["nrps"]["scopes"],
                )
                result = self.backend.apply_lti_roster(
                    auth,
                    course_id,
                    provider["id"],
                    service["contextId"],
                    roster["members"],
                )
                result["pages"] = roster["pages"]
                self.send_json(200, {"sync": result})
                return
            if path == "/api/courses/join":
                self.send_json(200, {"course": self.backend.join_course(auth, payload)})
                return
            match = re.fullmatch(r"/api/courses/([^/]+)/assignments", path)
            if match:
                self.send_json(
                    201,
                    {"assignment": self.backend.create_assignment(auth, match.group(1), payload)},
                )
                return
            match = re.fullmatch(r"/api/courses/([^/]+)/projects", path)
            if match:
                self.send_json(
                    201,
                    {"project": self.backend.save_project(auth, match.group(1), payload)},
                )
                return
            match = re.fullmatch(r"/api/assignments/([^/]+)/submissions", path)
            if match:
                self.send_json(
                    201,
                    {"submission": self.backend.submit_assignment(auth, match.group(1), payload)},
                )
                return
            match = re.fullmatch(r"/api/submissions/([^/]+)/grade", path)
            if match:
                self.send_json(
                    200,
                    {"submission": self.backend.grade_submission(auth, match.group(1), payload)},
                )
                return
            match = re.fullmatch(
                r"/api/submissions/([^/]+)/lti-grade-passback",
                path,
            )
            if match:
                plan = self.backend.prepare_lti_grade_passback(auth, match.group(1))
                context = plan["context"]
                lineitem_url = plan["lineitemUrl"]
                if not lineitem_url:
                    if not context.get("ags_lineitems_url"):
                        raise ApiError(
                            409,
                            "AGS line item service is unavailable",
                            "ags_lineitem_unavailable",
                        )
                    line_item = create_ags_line_item(
                        plan["provider"],
                        context["ags_lineitems_url"],
                        parse_json(context.get("ags_scope_json"), []),
                        plan["assignment"],
                    )
                    lineitem_url = self.backend.save_lti_line_item(
                        auth,
                        plan,
                        line_item["url"],
                    )
                else:
                    self.backend.save_lti_line_item(auth, plan, lineitem_url)
                result = post_ags_score(
                    plan["provider"],
                    lineitem_url,
                    parse_json(context.get("ags_scope_json"), []),
                    {
                        "userId": plan["studentSubject"],
                        "scoreGiven": plan["score"],
                        "scoreMaximum": 100,
                        "timestamp": iso_time(),
                        "comment": plan["feedback"],
                    },
                )
                self.send_json(
                    200,
                    {
                        "passback": self.backend.record_lti_grade_passback(
                            auth,
                            plan,
                            lineitem_url,
                            result,
                        )
                    },
                )
                return
            match = re.fullmatch(
                r"/api/courses/([^/]+)/members/([^/]+)/remove",
                path,
            )
            if match:
                self.send_json(
                    200,
                    self.backend.remove_course_member(
                        auth,
                        match.group(1),
                        match.group(2),
                    ),
                )
                return
            raise ApiError(404, "API endpoint was not found", "not_found")
        except ApiError as error:
            self.handle_api_error(error)
        except FederationError as error:
            self.handle_federation_error(error)
        except Exception as error:
            self.log_error("Unhandled API POST error: %s", error)
            self.send_json(500, {"error": {"code": "server_error", "message": "Internal server error"}})

    def do_PUT(self):
        path = self.api_path()
        try:
            match = re.fullmatch(r"/api/admin/identity-providers/([^/]+)", path)
            if not match:
                raise ApiError(404, "API endpoint was not found", "not_found")
            payload = self.read_json()
            auth = self.require_auth()
            self.require_csrf(auth)
            self.send_json(
                200,
                {
                    "provider": self.backend.update_identity_provider(
                        auth,
                        match.group(1),
                        payload,
                    )
                },
            )
        except ApiError as error:
            self.handle_api_error(error)
        except Exception as error:
            self.log_error("Unhandled API PUT error: %s", error)
            self.send_json(
                500,
                {"error": {"code": "server_error", "message": "Internal server error"}},
            )

    def log_message(self, format_string, *args):
        if self.api_path() not in ("/api/system-metrics", "/api/health"):
            super().log_message(format_string, *args)


def main():
    parser = argparse.ArgumentParser(description="Neural Blocks Lab local server")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), NeuralBlocksHandler)
    print(f"Neural Blocks Lab: http://{args.bind}:{args.port}")
    print(f"Classroom database: {BACKEND.database.description}")
    print("Authentication endpoints: /api/auth/register, /api/auth/login, /api/auth/me")
    print("System metrics endpoint: /api/system-metrics")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
