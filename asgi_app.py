import hashlib
import hmac
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from backend import ApiError, NeuralBlocksBackend
from federation import (
    FederationError,
    exchange_oidc_code,
    lti_authorization_url,
    oidc_authorization_url,
    validate_lti_claims,
    verify_id_token,
)
from job_queue import JobManager, QueuedMailer
from job_tasks import JobTaskRunner
from system_metrics import collect_metrics


ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE_TARGET = os.environ.get(
    "NBL_DATABASE_URL",
    os.environ.get("NBL_DATABASE_PATH", ROOT / ".data" / "neural_blocks.db"),
)
DEFAULT_BASE_URL = os.environ.get(
    "NBL_BASE_URL",
    "http://127.0.0.1:8770",
).rstrip("/")
MAX_JSON_BYTES = 2 * 1024 * 1024
AUTH_WINDOW_SECONDS = 15 * 60
AUTH_MAX_ATTEMPTS = 12
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "SAMEORIGIN",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "worker-src 'self' blob:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'self'"
    ),
}


def protected_static_path(path):
    decoded = unquote(path).replace("\\", "/")
    parts = [part for part in decoded.split("/") if part not in ("", ".")]
    if any(part.startswith(".") or part == "__pycache__" for part in parts):
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


def safe_return_path(value, base_url):
    value = str(value or "/").strip()
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        base = urlparse(base_url)
        if parsed.scheme != base.scheme or parsed.netloc != base.netloc:
            return "/"
        value = parsed.path or "/"
        if parsed.query:
            value += f"?{parsed.query}"
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    return value[:1000]


class AuthRateLimiter:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.attempts = {}
        self.lock = threading.Lock()

    def allow(self, ip_address, email):
        normalized = str(email or "").strip().lower()
        digest = hashlib.sha256(
            f"{ip_address}:{normalized}".encode("utf-8")
        ).hexdigest()
        if self.redis:
            key = f"neural-blocks:auth-rate:{digest}"
            try:
                value = self.redis.incr(key)
                if value == 1:
                    self.redis.expire(key, AUTH_WINDOW_SECONDS)
                return value <= AUTH_MAX_ATTEMPTS
            except Exception:
                pass
        now = time.time()
        with self.lock:
            recent = [
                attempt
                for attempt in self.attempts.get(digest, [])
                if now - attempt < AUTH_WINDOW_SECONDS
            ]
            if len(recent) >= AUTH_MAX_ATTEMPTS:
                self.attempts[digest] = recent
                return False
            recent.append(now)
            self.attempts[digest] = recent
            return True


def create_app(
    *,
    database_target=None,
    base_url=None,
    root=None,
    job_mode=None,
    redis_url=None,
    payload_key=None,
):
    root = Path(root or ROOT).resolve()
    database_target = database_target or DEFAULT_DATABASE_TARGET
    base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    expose_dev_tokens = os.environ.get("NBL_EXPOSE_DEV_TOKENS", "1") == "1"
    runner = JobTaskRunner(
        database_target,
        base_url=base_url,
        root=root,
    )
    jobs = JobManager(
        database_target,
        mode=job_mode,
        redis_url=redis_url,
        payload_key=payload_key,
        executor=runner,
    )
    backend = NeuralBlocksBackend(
        database_target,
        mailer=QueuedMailer(jobs),
        base_url=base_url,
        expose_dev_tokens=expose_dev_tokens,
    )
    limiter = AuthRateLimiter(jobs.redis)
    app = FastAPI(
        title="Neural Blocks Lab API",
        version="1.0.0",
        docs_url="/api/docs" if os.environ.get("NBL_API_DOCS") == "1" else None,
        redoc_url=None,
        openapi_url=(
            "/api/openapi.json"
            if os.environ.get("NBL_API_DOCS") == "1"
            else None
        ),
    )
    app.state.backend = backend
    app.state.jobs = jobs
    app.state.root = root
    app.state.base_url = base_url

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers[key] = value
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(_, error):
        return JSONResponse(
            status_code=error.status,
            content={"error": {"code": error.code, "message": error.message}},
        )

    @app.exception_handler(FederationError)
    async def federation_error_handler(_, error):
        return JSONResponse(
            status_code=error.status,
            content={"error": {"code": error.code, "message": error.message}},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_, error):
        print(f"Unhandled ASGI error: {error}", flush=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "server_error",
                    "message": "Internal server error",
                }
            },
        )

    async def call_backend(function, *args):
        return await run_in_threadpool(function, *args)

    async def json_body(request):
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            raise ApiError(
                415,
                "Content-Type must be application/json",
                "unsupported_media_type",
            )
        body = await request.body()
        if len(body) > MAX_JSON_BYTES:
            raise ApiError(413, "JSON request body is too large", "payload_too_large")
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError(400, "Request body must be valid JSON", "invalid_json")
        if not isinstance(payload, dict):
            raise ApiError(400, "JSON request body must be an object", "invalid_json")
        return payload

    async def form_body(request):
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" not in content_type:
            raise ApiError(
                415,
                "Content-Type must be application/x-www-form-urlencoded",
                "unsupported_media_type",
            )
        body = await request.body()
        if not body or len(body) > MAX_JSON_BYTES:
            raise ApiError(400, "Form request body is invalid", "invalid_request")
        try:
            values = parse_qs(
                body.decode("utf-8"),
                keep_blank_values=True,
                max_num_fields=50,
            )
        except (UnicodeDecodeError, ValueError):
            raise ApiError(400, "Form request body is invalid", "invalid_request")
        return {key: items[-1] for key, items in values.items()}

    def request_context(request):
        return {
            "ip": request.client.host if request.client else "",
            "userAgent": request.headers.get("user-agent", ""),
        }

    async def require_auth(request):
        auth = await call_backend(
            backend.authenticate,
            request.cookies.get("nbl_session"),
        )
        auth["_request"] = request_context(request)
        return auth

    def public_auth(auth):
        return {
            key: value
            for key, value in auth.items()
            if not str(key).startswith("_")
        }

    def require_csrf(request, auth):
        provided = request.headers.get("x-csrf-token", "")
        expected = auth.get("csrfToken", "")
        if not provided or not hmac.compare_digest(provided, expected):
            raise ApiError(403, "CSRF token is invalid", "csrf_failed")

    def set_session_cookie(response, token, clear=False):
        response.set_cookie(
            "nbl_session",
            "" if clear else token,
            path="/",
            httponly=True,
            secure=os.environ.get("NBL_SECURE_COOKIES") == "1",
            samesite="lax",
            max_age=0 if clear else 7 * 24 * 60 * 60,
        )

    @app.get("/api/health")
    async def health():
        queue_health = await run_in_threadpool(jobs.health)
        return {
            "status": (
                "ok"
                if queue_health["connected"] or queue_health["mode"] == "inline"
                else "degraded"
            ),
            "database": backend.database.description,
            "queue": queue_health,
            "server": "fastapi",
        }

    @app.get("/api/system-metrics")
    async def system_metrics():
        return await run_in_threadpool(collect_metrics)

    @app.get("/api/auth/providers")
    async def public_providers(tenant: str = ""):
        return {
            "providers": await call_backend(
                backend.public_identity_providers,
                tenant,
            )
        }

    @app.get("/api/auth/oidc/start")
    async def oidc_start(tenant: str, provider: str, returnTo: str = "/"):
        identity_provider = await call_backend(
            backend.get_identity_provider,
            provider,
            tenant,
            None,
            None,
            "oidc",
        )
        state = await call_backend(
            backend.create_federation_state,
            identity_provider,
            "oidc",
            safe_return_path(returnTo, base_url),
        )
        return RedirectResponse(
            oidc_authorization_url(
                identity_provider,
                state["state"],
                state["nonce"],
                f"{base_url}/api/auth/oidc/callback",
            ),
            status_code=302,
        )

    @app.get("/api/auth/oidc/callback")
    async def oidc_callback(request: Request, code: str = "", state: str = ""):
        if not code:
            raise ApiError(
                400,
                "OIDC authorization code is missing",
                "invalid_request",
            )
        state_data = await call_backend(
            backend.consume_federation_state,
            state,
            "oidc",
        )
        provider = state_data["provider"]
        token_response = await run_in_threadpool(
            exchange_oidc_code,
            provider,
            code,
            f"{base_url}/api/auth/oidc/callback",
        )
        claims = await run_in_threadpool(
            verify_id_token,
            provider,
            token_response["id_token"],
            state_data["nonce"],
        )
        auth = await call_backend(
            backend.resolve_federated_login,
            provider,
            claims,
            request_context(request),
        )
        token = auth.pop("sessionToken")
        response = RedirectResponse(
            safe_return_path(state_data["targetPath"], base_url),
            status_code=302,
        )
        set_session_cookie(response, token)
        return response

    @app.post("/api/auth/lti/login")
    async def lti_login(request: Request):
        form = await form_body(request)
        issuer = str(form.get("iss") or "").rstrip("/")
        provider = await call_backend(
            backend.get_identity_provider,
            None,
            None,
            issuer,
            form.get("client_id"),
            "lti",
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
        target_path = safe_return_path(
            form.get("target_link_uri") or "/",
            base_url,
        )
        state = await call_backend(
            backend.create_federation_state,
            provider,
            "lti",
            target_path,
            {
                "ltiMessageHint": form.get("lti_message_hint"),
                "deploymentId": deployment_id,
            },
        )
        return RedirectResponse(
            lti_authorization_url(
                provider,
                state["state"],
                state["nonce"],
                f"{base_url}/api/auth/lti/launch",
                login_hint,
                form.get("lti_message_hint"),
            ),
            status_code=302,
        )

    @app.post("/api/auth/lti/launch")
    async def lti_launch(request: Request):
        form = await form_body(request)
        state = await call_backend(
            backend.consume_federation_state,
            form.get("state"),
            "lti",
        )
        provider = state["provider"]
        claims = await run_in_threadpool(
            verify_id_token,
            provider,
            form.get("id_token"),
            state["nonce"],
        )
        validate_lti_claims(provider, claims)
        auth = await call_backend(
            backend.resolve_federated_login,
            provider,
            claims,
            request_context(request),
        )
        token = auth.pop("sessionToken")
        destination = safe_return_path(state["targetPath"], base_url)
        separator = "&" if "?" in destination else "?"
        response = RedirectResponse(
            f"{destination}{separator}lti=launched",
            status_code=302,
        )
        set_session_cookie(response, token)
        return response

    async def public_auth_action(request, action, status_code=200):
        payload = await json_body(request)
        if not limiter.allow(
            request.client.host if request.client else "",
            payload.get("email"),
        ):
            raise ApiError(429, "Too many authentication attempts", "rate_limited")
        result = await call_backend(action, payload, request_context(request))
        token = result.pop("sessionToken", None)
        response = JSONResponse(status_code=status_code, content=result)
        if token:
            set_session_cookie(response, token)
        return response

    @app.post("/api/auth/register")
    async def register(request: Request):
        return await public_auth_action(request, backend.register, 201)

    @app.post("/api/auth/login")
    async def login(request: Request):
        return await public_auth_action(request, backend.login)

    @app.post("/api/auth/invitations/accept")
    async def accept_invitation(request: Request):
        return await public_auth_action(
            request,
            backend.accept_invitation,
            201,
        )

    @app.post("/api/auth/verify-email")
    async def verify_email(request: Request):
        payload = await json_body(request)
        if not limiter.allow(
            request.client.host if request.client else "",
            "",
        ):
            raise ApiError(429, "Too many authentication attempts", "rate_limited")
        return await call_backend(
            backend.verify_email,
            payload,
            request_context(request),
        )

    @app.post("/api/auth/resend-verification", status_code=202)
    async def resend_verification(request: Request):
        payload = await json_body(request)
        if not limiter.allow(
            request.client.host if request.client else "",
            payload.get("email"),
        ):
            raise ApiError(429, "Too many authentication attempts", "rate_limited")
        return await call_backend(backend.resend_verification, payload)

    @app.post("/api/auth/password-reset/request", status_code=202)
    async def password_reset_request(request: Request):
        payload = await json_body(request)
        if not limiter.allow(
            request.client.host if request.client else "",
            payload.get("email"),
        ):
            raise ApiError(429, "Too many authentication attempts", "rate_limited")
        return await call_backend(backend.request_password_reset, payload)

    @app.post("/api/auth/password-reset/confirm")
    async def password_reset_confirm(request: Request):
        payload = await json_body(request)
        if not limiter.allow(
            request.client.host if request.client else "",
            "",
        ):
            raise ApiError(429, "Too many authentication attempts", "rate_limited")
        return await call_backend(
            backend.confirm_password_reset,
            payload,
            request_context(request),
        )

    @app.get("/api/auth/me")
    async def me(request: Request):
        return public_auth(await require_auth(request))

    @app.post("/api/auth/logout")
    async def logout(request: Request):
        auth = await require_auth(request)
        require_csrf(request, auth)
        await call_backend(
            backend.logout,
            request.cookies.get("nbl_session"),
        )
        response = JSONResponse({"status": "logged_out"})
        set_session_cookie(response, "", clear=True)
        return response

    @app.get("/api/admin/invitations")
    async def list_invitations(request: Request):
        auth = await require_auth(request)
        return {
            "invitations": await call_backend(
                backend.list_invitations,
                auth,
            )
        }

    @app.post("/api/admin/invitations", status_code=201)
    async def create_invitation(request: Request):
        payload = await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        return {
            "invitation": await call_backend(
                backend.create_invitation,
                auth,
                payload,
            )
        }

    @app.get("/api/admin/audit")
    async def list_audit(request: Request, limit: int = 100):
        auth = await require_auth(request)
        return {
            "events": await call_backend(
                backend.list_audit_events,
                auth,
                limit,
            )
        }

    @app.get("/api/admin/identity-providers")
    async def list_identity_providers(request: Request):
        auth = await require_auth(request)
        return {
            "providers": await call_backend(
                backend.list_identity_providers,
                auth,
            )
        }

    @app.post("/api/admin/identity-providers", status_code=201)
    async def create_identity_provider(request: Request):
        payload = await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        return {
            "provider": await call_backend(
                backend.create_identity_provider,
                auth,
                payload,
            )
        }

    @app.put("/api/admin/identity-providers/{provider_id}")
    async def update_identity_provider(
        provider_id: str,
        request: Request,
    ):
        payload = await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        return {
            "provider": await call_backend(
                backend.update_identity_provider,
                auth,
                provider_id,
                payload,
            )
        }

    @app.get("/api/courses")
    async def list_courses(request: Request):
        auth = await require_auth(request)
        return {"courses": await call_backend(backend.list_courses, auth)}

    @app.post("/api/courses", status_code=201)
    async def create_course(request: Request):
        payload = await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        return {
            "course": await call_backend(
                backend.create_course,
                auth,
                payload,
            )
        }

    @app.post("/api/courses/join")
    async def join_course(request: Request):
        payload = await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        return {
            "course": await call_backend(
                backend.join_course,
                auth,
                payload,
            )
        }

    @app.get("/api/courses/{course_id}/assignments")
    async def list_assignments(course_id: str, request: Request):
        auth = await require_auth(request)
        return {
            "assignments": await call_backend(
                backend.list_assignments,
                auth,
                course_id,
            )
        }

    @app.post("/api/courses/{course_id}/assignments", status_code=201)
    async def create_assignment(course_id: str, request: Request):
        payload = await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        return {
            "assignment": await call_backend(
                backend.create_assignment,
                auth,
                course_id,
                payload,
            )
        }

    @app.get("/api/courses/{course_id}/projects")
    async def list_projects(course_id: str, request: Request):
        auth = await require_auth(request)
        return {
            "projects": await call_backend(
                backend.list_projects,
                auth,
                course_id,
            )
        }

    @app.post("/api/courses/{course_id}/projects", status_code=201)
    async def save_project(course_id: str, request: Request):
        payload = await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        return {
            "project": await call_backend(
                backend.save_project,
                auth,
                course_id,
                payload,
            )
        }

    @app.get("/api/courses/{course_id}/submissions")
    async def list_submissions(course_id: str, request: Request):
        auth = await require_auth(request)
        return {
            "submissions": await call_backend(
                backend.list_submissions,
                auth,
                course_id,
            )
        }

    @app.get("/api/courses/{course_id}/members")
    async def list_course_members(course_id: str, request: Request):
        auth = await require_auth(request)
        return {
            "members": await call_backend(
                backend.list_course_members,
                auth,
                course_id,
            )
        }

    @app.post("/api/courses/{course_id}/members/{user_id}/remove")
    async def remove_course_member(
        course_id: str,
        user_id: str,
        request: Request,
    ):
        await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        return await call_backend(
            backend.remove_course_member,
            auth,
            course_id,
            user_id,
        )

    @app.get("/api/courses/{course_id}/lti-services")
    async def lti_course_service(course_id: str, request: Request):
        auth = await require_auth(request)
        return {
            "service": await call_backend(
                backend.get_lti_course_service,
                auth,
                course_id,
            )
        }

    @app.post("/api/courses/{course_id}/lti/roster-sync", status_code=202)
    async def queue_lti_roster(course_id: str, request: Request):
        await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        service = await call_backend(
            backend.get_lti_course_service,
            auth,
            course_id,
        )
        if not service["connected"] or not service["nrps"]["available"]:
            raise ApiError(
                409,
                "NRPS membership service is unavailable",
                "nrps_unavailable",
            )
        job = await run_in_threadpool(
            lambda: jobs.enqueue(
                "lti.roster_sync",
                {
                    "courseId": course_id,
                    "userId": auth["user"]["id"],
                },
                tenant_id=auth["user"]["tenantId"],
                user_id=auth["user"]["id"],
                dedupe_key=f"lti-roster:{course_id}",
                max_attempts=4,
            )
        )
        return {"job": job}

    @app.get("/api/projects/{project_id}")
    async def get_project(project_id: str, request: Request):
        auth = await require_auth(request)
        return {
            "project": await call_backend(
                backend.get_project,
                auth,
                project_id,
            )
        }

    @app.post("/api/assignments/{assignment_id}/submissions", status_code=201)
    async def submit_assignment(assignment_id: str, request: Request):
        payload = await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        return {
            "submission": await call_backend(
                backend.submit_assignment,
                auth,
                assignment_id,
                payload,
            )
        }

    @app.post("/api/submissions/{submission_id}/grade")
    async def grade_submission(submission_id: str, request: Request):
        payload = await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        return {
            "submission": await call_backend(
                backend.grade_submission,
                auth,
                submission_id,
                payload,
            )
        }

    @app.post(
        "/api/submissions/{submission_id}/lti-grade-passback",
        status_code=202,
    )
    async def queue_lti_grade(submission_id: str, request: Request):
        await json_body(request)
        auth = await require_auth(request)
        require_csrf(request, auth)
        await call_backend(
            backend.prepare_lti_grade_passback,
            auth,
            submission_id,
        )
        job = await run_in_threadpool(
            lambda: jobs.enqueue(
                "lti.grade_passback",
                {
                    "submissionId": submission_id,
                    "userId": auth["user"]["id"],
                },
                tenant_id=auth["user"]["tenantId"],
                user_id=auth["user"]["id"],
                dedupe_key=f"lti-grade:{submission_id}",
                max_attempts=5,
            )
        )
        return {"job": job}

    @app.get("/api/jobs")
    async def list_jobs(request: Request, limit: int = 30):
        auth = await require_auth(request)
        backend.require_role(auth, "admin", "professor")
        return {
            "jobs": await run_in_threadpool(
                jobs.list_for_auth,
                auth,
                limit,
            )
        }

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, request: Request):
        auth = await require_auth(request)
        backend.require_role(auth, "admin", "professor")
        job = await run_in_threadpool(jobs.get_for_auth, auth, job_id)
        if not job:
            raise ApiError(404, "Background job was not found", "not_found")
        return {"job": job}

    @app.api_route(
        "/{asset_path:path}",
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    async def static_asset(asset_path: str, request: Request):
        path = "/" + asset_path
        if path.startswith("/api/") or protected_static_path(path):
            return Response(status_code=404)
        requested = (root / (asset_path or "index.html")).resolve()
        try:
            requested.relative_to(root)
        except ValueError:
            return Response(status_code=404)
        if not requested.is_file():
            return Response(status_code=404)
        if request.method == "HEAD":
            return Response(
                status_code=200,
                media_type=None,
                headers={"Content-Length": str(requested.stat().st_size)},
            )
        return FileResponse(requested)

    return app


app = create_app()
